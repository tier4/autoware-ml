# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deployment script for exporting models to ONNX and TensorRT."""

import logging
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import onnx
import tensorrt as trt
import torch
from omegaconf import DictConfig, OmegaConf
from onnx.external_data_helper import convert_model_from_external_data
from torch.export import Dim

import autoware_ml.configs

logger = logging.getLogger(__name__)

_CONFIG_PATH = str(Path(autoware_ml.configs.__file__).parent.resolve())


def setup_logging(log_level: str) -> None:
    """Configure logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
    )


def validate_cuda_available() -> None:
    """Validate CUDA is available.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. TensorRT requires CUDA. "
            "Please run on a machine with CUDA support."
        )


def resolve_output_paths(
    checkpoint_path: Path, output_name: Optional[str], output_dir: Optional[str]
) -> tuple[Path, Path, Path]:
    """Resolve output paths for ONNX and TensorRT files.

    Args:
        checkpoint_path: Path to checkpoint file.
        output_name: Base name for output files (None to use checkpoint name).
        output_dir: Output directory (None to use checkpoint directory).

    Returns:
        Tuple of (output_dir, onnx_path, engine_path).
    """
    base_name = output_name if output_name else checkpoint_path.stem
    output_directory = Path(output_dir) if output_dir else checkpoint_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    onnx_path = output_directory / f"{base_name}.onnx"
    engine_path = output_directory / f"{base_name}.engine"

    return output_directory, onnx_path, engine_path


def load_model_from_checkpoint(
    model: L.LightningModule, checkpoint_path: Path, device: torch.device
) -> None:
    """Load model weights from checkpoint.

    Args:
        model: Model instance to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Device to load checkpoint on.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.to(device)


def extract_input_from_batch(batch: dict, param_name: str) -> torch.Tensor:
    """Extract input tensor from batch dictionary.

    Args:
        batch: Batch dictionary.
        param_name: Parameter name to extract.

    Returns:
        Input tensor.

    Raises:
        ValueError: If parameter not found in batch.
    """
    if param_name not in batch:
        raise ValueError(
            f"Parameter '{param_name}' not found in batch. Available keys: {list(batch.keys())}"
        )

    input_tensor = batch[param_name]

    if isinstance(input_tensor, (list, tuple)):
        input_tensor = input_tensor[0]

    return input_tensor


def get_input_sample(
    datamodule: L.LightningDataModule, model: L.LightningModule, device: torch.device
) -> torch.Tensor:
    """Get input sample from datamodule for model export.

    Args:
        datamodule: Lightning DataModule instance.
        model: Lightning Module instance.
        device: Device to move input sample to.

    Returns:
        Input tensor matching model's forward signature.

    Raises:
        ValueError: If model has no forward parameters or batch doesn't match.
    """
    datamodule.setup("predict")
    predict_dataloader = datamodule.predict_dataloader()
    batch = next(iter(predict_dataloader))

    forward_params = list(model.forward_signature.parameters.keys())
    if not forward_params:
        raise ValueError("Model forward signature has no parameters")

    param_name = forward_params[0]

    if isinstance(batch, dict):
        input_sample = extract_input_from_batch(batch, param_name)
    else:
        input_sample = batch
        if isinstance(input_sample, (list, tuple)):
            input_sample = input_sample[0]

    input_sample = input_sample.to(device)

    logger.info(f"Input sample shape: {input_sample.shape}")
    logger.info(f"Input sample dtype: {input_sample.dtype}")

    return input_sample


def build_dynamic_shapes(onnx_cfg: DictConfig, forward_params: list[str]) -> Optional[dict]:
    """Build dynamic shapes dictionary from config.

    Args:
        onnx_cfg: ONNX configuration.
        forward_params: List of forward parameter names.

    Returns:
        Dynamic shapes dictionary or None.
    """
    if "dynamic_shapes" not in onnx_cfg or onnx_cfg.dynamic_shapes is None:
        return None

    dynamic_shapes = {}
    for param_name, dim_mapping in onnx_cfg.dynamic_shapes.items():
        if param_name not in forward_params:
            logger.warning(
                f"Dynamic shape parameter '{param_name}' not found in forward signature. "
                f"Available parameters: {forward_params}. Skipping."
            )
            continue

        param_dynamic_shapes = {
            int(dim_idx): Dim(dim_name) for dim_idx, dim_name in dim_mapping.items()
        }

        if param_dynamic_shapes:
            dynamic_shapes[param_name] = param_dynamic_shapes

    return dynamic_shapes if dynamic_shapes else None


def merge_onnx_external_data(onnx_path: Path) -> None:
    """Merge external data files into a single ONNX file.

    Args:
        onnx_path: Path to ONNX file that may have external data.

    Raises:
        RuntimeError: If merging fails.
    """
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_from_external_data(onnx_model)
    onnx.save_model(onnx_model, str(onnx_path))


def export_to_onnx(
    model: L.LightningModule,
    input_sample: torch.Tensor,
    deploy_cfg: DictConfig,
    output_path: Path,
) -> None:
    """Export model to ONNX format.

    Args:
        model: Lightning Module instance.
        input_sample: Input tensor for export.
        deploy_cfg: Deployment configuration.
        output_path: Path to save ONNX file.

    Raises:
        ValueError: If model has no forward parameters.
        RuntimeError: If external data merging fails.
    """
    logger.info("Exporting model to ONNX...")

    onnx_cfg = deploy_cfg.onnx
    forward_params = list(model.forward_signature.parameters.keys())

    if not forward_params:
        raise ValueError("Model forward signature has no parameters")

    dynamic_shapes = build_dynamic_shapes(onnx_cfg, forward_params)

    logger.info(f"Dynamic shapes: {dynamic_shapes}")
    logger.info(f"ONNX opset version: {onnx_cfg.opset_version}")
    logger.info(f"Input names: {onnx_cfg.input_names}")
    logger.info(f"Output names: {onnx_cfg.output_names}")

    model.to_onnx(
        file_path=str(output_path),
        input_sample=input_sample,
        dynamic_shapes=dynamic_shapes,
        input_names=onnx_cfg.input_names,
        output_names=onnx_cfg.output_names,
        opset_version=onnx_cfg.opset_version,
        dynamo=True,
    )

    logger.info(f"Successfully exported ONNX model to {output_path}")

    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        logger.info(f"Found external data file: {data_path}. Merging into ONNX file...")
        merge_onnx_external_data(output_path)
        data_path.unlink()
        logger.info("Successfully merged external data into ONNX file")


def instantiate_modifier(modify_graph_cfg: DictConfig):
    """Instantiate ONNX graph modifier from config.

    Args:
        modify_graph_cfg: Configuration for graph modification.

    Returns:
        Instantiated modifier (callable or object with modify method).

    Raises:
        ValueError: If modifier is not callable or has no modify method.
    """
    modifier = hydra.utils.instantiate(modify_graph_cfg)

    if callable(modifier):
        return modifier

    if hasattr(modifier, "modify"):
        return modifier

    raise ValueError(f"Modifier {modifier} must be callable or have a 'modify' method")


def apply_modifier(modifier, onnx_path: Path) -> Path:
    """Apply modifier to ONNX graph.

    Args:
        modifier: Modifier instance (callable or has modify method).
        onnx_path: Path to ONNX file.

    Returns:
        Path to modified ONNX file.

    Raises:
        ValueError: If modifier returns None or invalid path.
    """
    if callable(modifier):
        modified_path = modifier(onnx_path)
    else:
        modified_path = modifier.modify(onnx_path)

    if modified_path is None:
        raise ValueError("Modifier returned None. Must return Path or str.")

    return Path(modified_path)


def should_modify_graph(modify_graph_cfg: Optional[DictConfig]) -> bool:
    """Check if ONNX graph modification is configured.

    Args:
        modify_graph_cfg: Configuration for graph modification (may be None).

    Returns:
        True if modification should be applied, False otherwise.
    """
    if modify_graph_cfg is None:
        return False

    if isinstance(modify_graph_cfg, DictConfig) and OmegaConf.is_none(modify_graph_cfg):
        return False

    return True


def modify_onnx_graph(onnx_path: Path, modify_graph_cfg: DictConfig) -> Path:
    """Modify ONNX graph using modifier specified in config.

    Args:
        onnx_path: Path to ONNX file.
        modify_graph_cfg: Configuration for graph modification.

    Returns:
        Path to modified ONNX file.

    Raises:
        ValueError: If modifier is invalid or returns invalid path.
    """
    logger.info("Modifying ONNX graph...")

    modifier = instantiate_modifier(modify_graph_cfg)
    modified_path = apply_modifier(modifier, onnx_path)

    logger.info(f"Successfully modified ONNX graph: {modified_path}")
    return modified_path


def create_tensorrt_builder_config(tensorrt_cfg: DictConfig) -> tuple:
    """Create TensorRT builder and configuration.

    Args:
        tensorrt_cfg: TensorRT configuration.

    Returns:
        Tuple of (builder, network, parser, config).
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    workspace_size = tensorrt_cfg.get("workspace_size", 1 << 30)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    logger.info(f"Workspace size: {workspace_size / (1024**3):.2f} GB")

    return builder, network, parser, config


def parse_onnx_file(parser, onnx_path: Path) -> None:
    """Parse ONNX file with TensorRT parser.

    Args:
        parser: TensorRT ONNX parser.
        onnx_path: Path to ONNX file.

    Raises:
        RuntimeError: If parsing fails.
    """
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        error_msg = "\n".join(f"TensorRT parser error {i}: {err}" for i, err in enumerate(errors))
        raise RuntimeError(f"Failed to parse ONNX file:\n{error_msg}")

    logger.info("Successfully parsed ONNX file")


def create_optimization_profile(
    builder: "trt.Builder", tensorrt_cfg: DictConfig
) -> Optional["trt.IOptimizationProfile"]:
    """Create TensorRT optimization profile from config.

    Args:
        builder: TensorRT builder.
        tensorrt_cfg: TensorRT configuration.

    Returns:
        Optimization profile or None if not configured.
    """
    if "input_shapes" not in tensorrt_cfg:
        return None

    profile = builder.create_optimization_profile()

    for input_name, shapes in tensorrt_cfg.input_shapes.items():
        min_shape = shapes.get("min_shape")
        opt_shape = shapes.get("opt_shape")
        max_shape = shapes.get("max_shape")

        if not (min_shape and opt_shape and max_shape):
            continue

        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        logger.info(
            f"Optimization profile for '{input_name}': "
            f"min={min_shape}, opt={opt_shape}, max={max_shape}"
        )

    return profile


def build_tensorrt_engine(
    onnx_path: Path,
    deploy_cfg: DictConfig,
    output_path: Path,
) -> None:
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX file.
        deploy_cfg: Deployment configuration.
        output_path: Path to save TensorRT engine.

    Raises:
        RuntimeError: If engine building fails.
    """
    logger.info("Building TensorRT engine...")

    tensorrt_cfg = deploy_cfg.tensorrt
    builder, network, parser, config = create_tensorrt_builder_config(tensorrt_cfg)

    parse_onnx_file(parser, onnx_path)

    profile = create_optimization_profile(builder, tensorrt_cfg)
    if profile is not None:
        config.add_optimization_profile(profile)

    logger.info("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"Successfully built TensorRT engine: {output_path}")


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Main deployment function.

    Args:
        cfg: Hydra configuration.
    """
    if "checkpoint" not in cfg:
        raise ValueError("Checkpoint must be specified (e.g., +checkpoint=path/to/checkpoint.ckpt)")

    if "deploy" not in cfg:
        raise ValueError("Config must have 'deploy' section")

    log_level = cfg.get("log_level", "INFO")
    setup_logging(log_level)

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    validate_cuda_available()

    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    output_name = cfg.get("output_name", None)
    output_dir = cfg.get("output_dir", None)
    output_dir, onnx_path, engine_path = resolve_output_paths(
        checkpoint_path, output_name, output_dir
    )

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"ONNX output: {onnx_path}")
    logger.info(f"TensorRT engine output: {engine_path}")

    deploy_cfg = cfg.deploy

    logger.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    logger.info("Instantiating model from config...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
    load_model_from_checkpoint(model, checkpoint_path, device)

    logger.info("Getting input sample...")
    input_sample = get_input_sample(datamodule, model, device)

    logger.info("Exporting to ONNX...")
    export_to_onnx(model, input_sample, deploy_cfg, onnx_path)

    modify_graph_cfg = deploy_cfg.onnx.get("modify_graph", None)
    if should_modify_graph(modify_graph_cfg):
        onnx_path = modify_onnx_graph(onnx_path, modify_graph_cfg)

    logger.info("Building TensorRT engine...")
    build_tensorrt_engine(onnx_path, deploy_cfg, engine_path)

    logger.info("Deployment completed successfully!")
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"TensorRT engine: {engine_path}")


if __name__ == "__main__":
    main()
