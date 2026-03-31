# Copyright 2026 TIER IV, Inc.
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

"""Deployment utility types and helpers.

This module defines the canonical export contract used by deployment code:

- models with standard tensor-only forwards rely on the generic export path
- models with special export requirements override ``build_export_spec(batch)``
  and return :class:`ExportSpec`
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from typing import Any

import lightning as L
from omegaconf import DictConfig, OmegaConf
import torch
from torch.export import Dim

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportSpec:
    """Describe the module and tensor inputs used for model export.

    Attributes:
        module: Module instance exported to ONNX.
        args: Example positional inputs supplied during export.
        input_param_names: Names associated with the positional input tensors.
        output_names: Optional names associated with exported output tensors.
    """

    module: torch.nn.Module
    args: tuple[Any, ...]
    input_param_names: list[str]
    output_names: list[str] | None = None


def validate_cuda_available() -> None:
    """Ensure CUDA is available for deployment export."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. TensorRT requires CUDA. "
            "Please run on a machine with CUDA support."
        )


def resolve_output_paths(
    checkpoint_path: Path,
    output_name: str | None,
    output_dir: str | None,
) -> tuple[Path, Path, Path]:
    """Resolve the output directory and export artifact paths."""
    base_name = output_name if output_name else checkpoint_path.stem
    output_directory = Path(output_dir) if output_dir else checkpoint_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    onnx_path = output_directory / f"{base_name}.onnx"
    engine_path = output_directory / f"{base_name}.engine"
    return output_directory, onnx_path, engine_path


def get_forward_signature(model: L.LightningModule) -> inspect.Signature:
    """Return the cached forward signature from BaseModel, or compute it."""
    return getattr(model, "forward_signature", inspect.signature(model.forward))


def get_export_parameter_names(model: L.LightningModule) -> list[str]:
    """Return concrete forward parameter names used for export."""
    signature = get_forward_signature(model)
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors nested in common Python containers to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def extract_input_from_batch(batch: dict[str, Any], param_name: str) -> Any:
    """Extract one export input from a batch dictionary."""
    if param_name not in batch:
        raise ValueError(
            f"Parameter '{param_name}' not found in batch. Available keys: {list(batch.keys())}"
        )

    input_value = batch[param_name]
    if isinstance(input_value, (list, tuple)):
        input_value = input_value[0]
    return input_value


def get_predict_batch(datamodule: L.LightningDataModule, device: torch.device) -> dict[str, Any]:
    """Load one prediction batch and apply transfer-time preprocessing."""
    datamodule.setup("predict")
    predict_dataloader = datamodule.predict_dataloader()
    batch = next(iter(predict_dataloader))
    batch = move_to_device(batch, device)

    if getattr(datamodule, "data_preprocessing", None) is not None:
        datamodule.data_preprocessing.to(device)
    return datamodule.on_after_batch_transfer(batch, dataloader_idx=0)


def infer_export_spec(model: L.LightningModule, batch: dict[str, Any]) -> ExportSpec:
    """Infer an export specification directly from the model forward signature."""
    forward_params = get_export_parameter_names(model)
    if not forward_params:
        raise ValueError("Model forward signature has no parameters.")

    if (
        isinstance(batch, dict)
        and len(forward_params) == 1
        and forward_params[0] == "batch_inputs_dict"
    ):
        return ExportSpec(module=model, args=(batch,), input_param_names=forward_params)

    input_args = tuple(extract_input_from_batch(batch, param_name) for param_name in forward_params)
    return ExportSpec(module=model, args=input_args, input_param_names=forward_params)


def resolve_export_spec(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
) -> ExportSpec:
    """Resolve the effective export specification for a model.

    Models with deployment-specific wrappers override ``build_export_spec``
    on :class:`~autoware_ml.models.base.BaseModel`. The default implementation
    falls back to the generic forward-signature-based export path.
    """
    batch = get_predict_batch(datamodule, device)
    return model.build_export_spec(batch)


def log_export_inputs(input_args: tuple[Any, ...], input_names: list[str]) -> None:
    """Log export input metadata for debugging."""
    for input_name, input_value in zip(input_names, input_args):
        if isinstance(input_value, torch.Tensor):
            logger.info(
                "Input '%s': shape=%s, dtype=%s",
                input_name,
                tuple(input_value.shape),
                input_value.dtype,
            )
        else:
            logger.info("Input '%s': type=%s", input_name, type(input_value).__name__)


def build_dynamic_shapes(
    onnx_cfg: DictConfig,
    forward_params: list[str],
) -> dict[str, dict[int, Dim]] | None:
    """Build the ONNX dynamic-shape mapping from config."""
    if "dynamic_shapes" not in onnx_cfg or onnx_cfg.dynamic_shapes is None:
        return None

    dynamic_shapes: dict[str, dict[int, Dim]] = {}
    for param_name, dim_mapping in onnx_cfg.dynamic_shapes.items():
        if param_name not in forward_params:
            logger.warning(
                "Dynamic shape parameter '%s' not found in forward signature. Available parameters: %s. Skipping.",
                param_name,
                forward_params,
            )
            continue

        param_dynamic_shapes: dict[int, Dim] = {}
        for dim_idx, dim_spec in dim_mapping.items():
            if isinstance(dim_spec, str):
                param_dynamic_shapes[int(dim_idx)] = Dim(dim_spec)
                continue

            dim_name = dim_spec.get("name")
            if dim_name is None:
                raise ValueError(
                    f"Dynamic shape spec for '{param_name}[{dim_idx}]' must define 'name'."
                )
            dim_kwargs = {key: dim_spec[key] for key in ("min", "max") if key in dim_spec}
            param_dynamic_shapes[int(dim_idx)] = Dim(dim_name, **dim_kwargs)

        if param_dynamic_shapes:
            dynamic_shapes[param_name] = param_dynamic_shapes

    return dynamic_shapes or None


def merge_onnx_external_data(onnx_path: Path) -> None:
    """Merge ONNX external data shards back into a single file."""
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_from_external_data(onnx_model)
    onnx.save_model(onnx_model, str(onnx_path))


def export_to_onnx(
    model: torch.nn.Module,
    input_sample: tuple[Any, ...],
    deploy_cfg: DictConfig,
    input_param_names: list[str],
    output_names_override: list[str] | None,
    output_path: Path,
) -> None:
    """Export a model to ONNX."""
    logger.info("Exporting model to ONNX...")

    onnx_cfg = deploy_cfg.onnx
    if not input_param_names:
        raise ValueError("Model forward signature has no parameters.")

    dynamic_shapes = build_dynamic_shapes(onnx_cfg, input_param_names)
    input_names = list(onnx_cfg.get("input_names", input_param_names))
    output_names = list(output_names_override or onnx_cfg.get("output_names", ["output"]))

    logger.info("Dynamic shapes: %s", dynamic_shapes)
    logger.info("ONNX opset version: %s", onnx_cfg.opset_version)
    logger.info("Input names: %s", input_names)
    logger.info("Output names: %s", output_names)
    log_export_inputs(input_sample, input_param_names)

    torch.onnx.export(
        model=model,
        args=input_sample,
        f=str(output_path),
        dynamic_shapes=dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
        opset_version=onnx_cfg.opset_version,
        dynamo=onnx_cfg.get("dynamo", True),
        do_constant_folding=onnx_cfg.get("do_constant_folding", True),
    )

    logger.info("Successfully exported ONNX model to %s", output_path)

    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        logger.info("Found external data file %s. Merging into the ONNX file...", data_path)
        merge_onnx_external_data(output_path)
        data_path.unlink()
        logger.info("Successfully merged external data into the ONNX file")


def instantiate_modifier(modify_graph_cfg: DictConfig) -> Any:
    """Instantiate an ONNX graph modifier from config."""
    import hydra

    modifier = hydra.utils.instantiate(modify_graph_cfg)
    if callable(modifier):
        return modifier
    if hasattr(modifier, "modify"):
        return modifier
    raise ValueError(f"Modifier {modifier} must be callable or have a 'modify' method.")


def apply_modifier(modifier: Any, onnx_path: Path) -> Path:
    """Apply a configured ONNX graph modifier."""
    modified_path = modifier(onnx_path) if callable(modifier) else modifier.modify(onnx_path)
    if modified_path is None:
        raise ValueError("Modifier returned None. Must return Path or str.")
    return Path(modified_path)


def should_modify_graph(modify_graph_cfg: DictConfig | None) -> bool:
    """Return whether graph modification is enabled."""
    if modify_graph_cfg is None:
        return False
    if isinstance(modify_graph_cfg, DictConfig) and OmegaConf.is_none(modify_graph_cfg):
        return False
    return True


def modify_onnx_graph(onnx_path: Path, modify_graph_cfg: DictConfig) -> Path:
    """Modify an ONNX graph using the configured modifier."""
    logger.info("Modifying ONNX graph...")
    modifier = instantiate_modifier(modify_graph_cfg)
    modified_path = apply_modifier(modifier, onnx_path)
    logger.info("Successfully modified ONNX graph: %s", modified_path)
    return modified_path


def create_tensorrt_builder_config(tensorrt_cfg: DictConfig) -> tuple[Any, Any, Any, Any]:
    """Create TensorRT builder objects for engine generation."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()

    workspace_size = tensorrt_cfg.get("workspace_size", 1 << 30)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    logger.info("Workspace size: %.2f GB", workspace_size / (1024**3))
    return builder, network, parser, config


def parse_onnx_file(parser: Any, onnx_path: Path) -> None:
    """Parse an ONNX file with a TensorRT parser."""
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        error_msg = "\n".join(f"TensorRT parser error {i}: {err}" for i, err in enumerate(errors))
        raise RuntimeError(f"Failed to parse ONNX file:\n{error_msg}")

    logger.info("Successfully parsed ONNX file")


def create_optimization_profile(builder: Any, tensorrt_cfg: DictConfig) -> Any | None:
    """Create a TensorRT optimization profile from config."""
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
            "Optimization profile for '%s': min=%s, opt=%s, max=%s",
            input_name,
            min_shape,
            opt_shape,
            max_shape,
        )
    return profile


def build_tensorrt_engine(
    onnx_path: Path,
    deploy_cfg: DictConfig,
    output_path: Path,
) -> None:
    """Build a TensorRT engine from an ONNX model."""
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
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("Successfully built TensorRT engine: %s", output_path)


def should_export_stage(stage_cfg: DictConfig | None) -> bool:
    """Return whether an export stage is enabled."""
    if stage_cfg is None:
        return False
    return bool(stage_cfg.get("enabled", True))
