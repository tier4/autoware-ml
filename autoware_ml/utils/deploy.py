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

This module defines the canonical per-module export contract used by
deployment code. Models expose ``build_export_specs(batch)`` and return a
mapping from module names to :class:`ExportSpec` objects.
"""

from __future__ import annotations

from collections.abc import Iterable

from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import onnx
from onnx.external_data_helper import convert_model_from_external_data
from omegaconf import DictConfig, OmegaConf
import torch
from torch.export import Dim

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.ops.segment.scatter_reduce import register_scatter_reduce_onnx_symbolic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportSpec:
    """Describe the module and tensor inputs used for model export.

    Attributes:
        module: Module instance exported to ONNX.
        args: Example positional inputs supplied during export.
        input_names: ONNX input names, one per entry of ``args`` in order; the keys of
            ``deploy.tensorrt.modules.<module>.input_shapes`` and the names the runtime binds.
        output_names: Names of the exported output tensors, in order.
        dynamic_axes: The axes that vary between runs, ``{tensor: {dim: name}}``, keyed by
            the input and output names. The model declares them here, not in its config.
            The TorchScript exporter (``dynamo=False``) takes them as its ``dynamic_axes``;
            ``torch.export`` (``dynamo=True``) gets one ``Dim`` per name over the inputs.
        supported_stages: Export stages supported by this specification.
    """

    module: torch.nn.Module
    args: tuple[Any, ...]
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]] | None = None
    supported_stages: frozenset[str] = frozenset({"onnx", "tensorrt"})

    def __post_init__(self) -> None:
        """Refuse a spec that leaves an input or every output unnamed, or an axis of no tensor.

        Checked when the spec is built, so a model fails before any of its modules exports.

        Raises:
            ValueError: If ``input_names`` is empty or not one name per entry of ``args``,
                ``output_names`` is empty, or ``dynamic_axes`` names a tensor that is neither
                an input nor an output.
        """
        if not self.input_names or len(self.input_names) != len(self.args):
            raise ValueError(
                f"The export spec names {len(self.input_names)} inputs for {len(self.args)} args."
            )
        if not self.output_names:
            raise ValueError("The export spec names no outputs.")
        # The TorchScript exporter only warns about an axis of an unknown tensor and exports
        # it static.
        unknown = sorted(
            set(self.dynamic_axes or ()) - set(self.input_names) - set(self.output_names)
        )
        if unknown:
            raise ValueError(
                f"The export spec declares dynamic_axes for {unknown}, which are not among its "
                f"inputs {self.input_names} or outputs {self.output_names}."
            )


def validate_cuda_available() -> None:
    """Ensure CUDA is available for deployment export."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. TensorRT requires CUDA. "
            "Please run on a machine with CUDA support."
        )


def get_predict_batch(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
) -> ModelBatchInputs:
    """Load one prediction batch and apply transfer-time preprocessing."""
    datamodule.prepare_data()
    datamodule.setup("predict")
    predict_dataloader = datamodule.predict_dataloader()
    # The typed batch moves its own tensors
    batch = next(iter(predict_dataloader)).to_device(device)
    return model.on_after_batch_transfer(batch, dataloader_idx=0)


def resolve_export_specs(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
) -> dict[str, ExportSpec]:
    """Resolve per-module export specifications for a model.

    Args:
        datamodule: Data module used to generate one prediction batch.
        model: Model instance to export.
        device: Device for tensor operations during export preparation.

    Returns:
        Ordered mapping of module name to export specification.
    """
    batch = get_predict_batch(datamodule, model, device)
    return model.build_export_specs(batch)


def merge_module_onnx_cfg(onnx_cfg: DictConfig, module_name: str) -> DictConfig:
    """Return the ONNX settings of one exported module.

    The shared ``deploy.onnx`` settings, overridden by the module's entry under
    ``deploy.onnx.modules``. The ``modules`` key itself is left out.

    Args:
        onnx_cfg: ``deploy.onnx``, with one ``modules`` entry per exported module.
        module_name: The exported module, as the export specs name it.

    Raises:
        KeyError: When the module has no entry; an empty entry keeps the shared settings.
    """
    modules_cfg = onnx_cfg.get("modules") or {}
    if module_name not in modules_cfg:
        raise KeyError(
            f"deploy.onnx.modules has no entry for exported module '{module_name}'; "
            f"its entries are {sorted(modules_cfg)}. Add '{module_name}: {{}}' to export it "
            "with the shared deploy.onnx settings."
        )
    shared = {
        k: v for k, v in OmegaConf.to_container(onnx_cfg, resolve=True).items() if k != "modules"
    }
    module_cfg = modules_cfg[module_name]
    module_overrides = OmegaConf.to_container(module_cfg, resolve=True) if module_cfg else {}
    return OmegaConf.create({**shared, **module_overrides})


def validate_onnx_modules_exported(onnx_cfg: DictConfig, module_names: Iterable[str]) -> None:
    """Refuse ``deploy.onnx.modules`` that lacks an entry of a module the model exports.

    Every exported module needs an entry, ``{}`` when it keeps the shared settings, so a module
    added to the model cannot silently export without its ``metainfo`` or ``modify_graph``, and a
    misspelled entry shows up as the missing one. Entries of modules the model does not export
    are left alone: a model whose exported modules depend on its config, as BEVFusion's on
    ``camera_network``, lists the entries of every variant in one base config.

    Args:
        onnx_cfg: ``deploy.onnx``, with one ``modules`` entry per exported module.
        module_names: The modules the export specs name.

    Raises:
        ValueError: Naming the modules without an entry and the entries there are.
    """
    module_names = list(module_names)
    entries = sorted(onnx_cfg.get("modules") or {})
    missing = sorted(set(module_names) - set(entries))
    if missing:
        raise ValueError(
            f"deploy.onnx.modules has no entry for the exported modules {missing}; its entries "
            f"are {entries} and the model exports {module_names}. An empty entry ({{}}) keeps "
            "the shared deploy.onnx settings."
        )


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


def dynamic_shapes_from_spec(spec: ExportSpec) -> tuple[dict[int, Dim] | None, ...] | None:
    """Return the ``torch.export`` dynamic shapes of the spec's inputs, in ``args`` order.

    One ``Dim`` per axis name, shared by every axis that carries the name, so equal axes
    stay equal in the exported graph. Output axes follow from the inputs and are left out.

    Returns:
        One mapping per input, ``None`` for a static input, or ``None`` when no input has a
        dynamic axis.
    """
    if not spec.dynamic_axes:
        return None
    dims: dict[str, Dim] = {}
    shapes = []
    for input_name in spec.input_names:
        axes = spec.dynamic_axes.get(input_name)
        shapes.append(
            {index: dims.setdefault(name, Dim(name)) for index, name in axes.items()}
            if axes
            else None
        )
    if all(shape is None for shape in shapes):
        return None
    return tuple(shapes)


def normalize_dynamic_shapes_for_model(
    model: torch.nn.Module,
    dynamic_shapes: tuple[dict[int, Dim] | None, ...] | None,
) -> tuple[Any, ...] | None:
    """Adapt dynamic-shape structure to the model forward signature.

    ``torch.export`` requires ``dynamic_shapes`` to mirror the positional input
    pytree passed to the model. Wrappers that expose ``forward(*args)`` receive
    one tuple-valued positional argument, so their dynamic-shape structure must
    be wrapped one level deeper.
    """
    if dynamic_shapes is None:
        return None

    signature = inspect.signature(model.forward)
    parameters = [parameter for parameter in signature.parameters.values()]
    if len(parameters) == 1 and parameters[0].kind == inspect.Parameter.VAR_POSITIONAL:
        return (dynamic_shapes,)
    return dynamic_shapes


def merge_onnx_external_data(onnx_path: Path) -> None:
    """Merge ONNX external data shards back into a single file."""
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_from_external_data(onnx_model)
    onnx.save_model(onnx_model, str(onnx_path))


def validate_onnx_names_not_configured(onnx_cfg: DictConfig) -> None:
    """Refuse the tensor names and the dynamic axes or shapes in ``deploy.onnx``.

    The export spec names the tensors and declares the dynamic axes, for both exporters.

    ``run_deploy`` checks it before the MLflow run is created or the model loads.

    Args:
        onnx_cfg: ``deploy.onnx``, with its optional ``modules`` overrides.

    Raises:
        ValueError: If the shared settings or a module's settings carry either key.
    """
    scopes = {"deploy.onnx": onnx_cfg}
    for module_name, module_cfg in (onnx_cfg.get("modules") or {}).items():
        scopes[f"deploy.onnx.modules.{module_name}"] = module_cfg
    configured = [
        f"{scope}.{key}"
        for scope, cfg in scopes.items()
        for key in ("input_names", "output_names", "dynamic_axes", "dynamic_shapes")
        if cfg is not None and key in cfg
    ]
    if configured:
        raise ValueError(
            f"{configured} are not read: the export spec names the graph inputs and outputs "
            "and declares the dynamic axes. Remove the entries."
        )


def export_to_onnx(spec: ExportSpec, onnx_cfg: DictConfig, output_path: Path) -> None:
    """Export a spec's module to ONNX under the spec's input and output names.

    Args:
        spec: The module, its example inputs and the tensor names.
        onnx_cfg: The module's merged ``deploy.onnx`` settings.
        output_path: Where the ``.onnx`` file goes.
    """
    logger.info("Exporting model to ONNX...")

    dynamo = onnx_cfg.get("dynamo", True)
    dynamic_shapes = (
        normalize_dynamic_shapes_for_model(spec.module, dynamic_shapes_from_spec(spec))
        if dynamo
        else None
    )
    dynamic_axes = None if dynamo else spec.dynamic_axes

    logger.info("Dynamic shapes: %s", dynamic_shapes)
    logger.info("Dynamic axes: %s", dynamic_axes)
    logger.info("ONNX opset version: %s", onnx_cfg.opset_version)
    logger.info("Input names: %s", spec.input_names)
    logger.info("Output names: %s", spec.output_names)
    log_export_inputs(spec.args, spec.input_names)

    # Register shared ONNX symbolics needed by export-aware ops packages.
    register_scatter_reduce_onnx_symbolic(opset_version=int(onnx_cfg.opset_version))

    export_kwargs = {
        "model": spec.module,
        "args": spec.args,
        "f": str(output_path),
        "input_names": spec.input_names,
        "output_names": spec.output_names,
        "opset_version": onnx_cfg.opset_version,
        "dynamo": dynamo,
        "do_constant_folding": onnx_cfg.get("do_constant_folding", True),
    }
    if dynamo:
        export_kwargs["dynamic_shapes"] = dynamic_shapes
    else:
        export_kwargs["dynamic_axes"] = dynamic_axes

    torch.onnx.export(**export_kwargs)

    logger.info("Successfully exported ONNX model to %s", output_path)

    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        logger.info("Found external data file %s. Merging into the ONNX file...", data_path)
        merge_onnx_external_data(output_path)
        data_path.unlink()
        logger.info("Successfully merged external data into the ONNX file")


def instantiate_modifier(modify_graph_cfg: DictConfig) -> Any:
    """Instantiate an ONNX graph modifier from config."""
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
    if isinstance(modify_graph_cfg, DictConfig):
        return OmegaConf.to_container(modify_graph_cfg, resolve=False) is not None
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
    # TODO(JonnaMat): Import tensorrt at the top.
    import tensorrt as trt  # noqa: PLC0415

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    builder = trt.Builder(trt_logger)
    # Always strongly typed: deploy.onnx.precision decides which dtypes the ONNX carries, and the
    # engine has to use them as exported rather than let the builder reassign precisions.
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


def create_optimization_profile(
    builder: Any, tensorrt_cfg: DictConfig, network: Any, module_name: str
) -> Any | None:
    """Create the optimization profile of ``network`` from its module's ``input_shapes``.

    The shapes live under ``deploy.tensorrt.modules.<module_name>.input_shapes``, keyed by
    the module's ONNX input names, next to the module's ``deploy.onnx.modules`` entry. A
    static input needs no entry. A dynamic input without one, or an entry that names no input
    of the network (a misspelled key), is refused here, naming the config key: TensorRT would
    refuse the build later with a message that does not, or silently build without the entry.

    Returns:
        The profile, or ``None`` when the module has no entry.

    Raises:
        ValueError: When the table is at the old place (``deploy.tensorrt.input_shapes``), an
            entry names no input of the network, a dynamic input has no entry, or an entry
            lacks one of its three shapes.
    """
    key = f"deploy.tensorrt.modules.{module_name}.input_shapes"
    if "input_shapes" in tensorrt_cfg:
        raise ValueError(
            "deploy.tensorrt.input_shapes is not read: the shapes of each exported module live "
            f"under deploy.tensorrt.modules.<module>.input_shapes. Move the entries to {key}."
        )
    module_cfg = (tensorrt_cfg.get("modules") or {}).get(module_name) or {}
    input_shapes = (
        OmegaConf.to_container(module_cfg.input_shapes, resolve=True)
        if "input_shapes" in module_cfg
        else {}
    )
    inputs = [network.get_input(index) for index in range(network.num_inputs)]
    unknown = sorted(set(input_shapes) - {tensor.name for tensor in inputs})
    if unknown:
        raise ValueError(
            f"{key} names {unknown}, which are not inputs of the {module_name!r} graph "
            f"{[tensor.name for tensor in inputs]}."
        )

    profile = None
    missing = []
    for tensor in inputs:
        shapes = input_shapes.get(tensor.name)
        if shapes is None:
            if -1 in tuple(tensor.shape):
                missing.append(tensor.name)
            continue
        min_shape = shapes.get("min_shape")
        opt_shape = shapes.get("opt_shape")
        max_shape = shapes.get("max_shape")
        if not (min_shape and opt_shape and max_shape):
            raise ValueError(
                f"{key}.{tensor.name} is incomplete. All of min_shape, opt_shape, and "
                "max_shape must be specified."
            )
        if profile is None:
            profile = builder.create_optimization_profile()
        profile.set_shape(tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        logger.info(
            "Optimization profile for '%s': min=%s, opt=%s, max=%s",
            tensor.name,
            min_shape,
            opt_shape,
            max_shape,
        )
    if missing:
        raise ValueError(
            f"The dynamic inputs {missing} of the {module_name!r} graph have no {key} entry; "
            "TensorRT needs a min / opt / max shape for each."
        )
    return profile


def build_tensorrt_engine(
    onnx_path: Path,
    deploy_cfg: DictConfig,
    output_path: Path,
    module_name: str,
) -> None:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: The module's ONNX file.
        deploy_cfg: The ``deploy`` section; the module's shapes are read from
            ``deploy.tensorrt.modules.<module_name>.input_shapes``.
        output_path: Where the ``.engine`` file goes.
        module_name: The exported module, as the export specs name it.

    Raises:
        ValueError: When the module's ``input_shapes`` entry is wrong or missing for a dynamic
            input (see :func:`create_optimization_profile`).
        RuntimeError: When the build fails.
    """
    logger.info("Building TensorRT engine...")
    tensorrt_cfg = deploy_cfg.tensorrt
    builder, network, parser, config = create_tensorrt_builder_config(tensorrt_cfg)
    parse_onnx_file(parser, onnx_path)

    profile = create_optimization_profile(builder, tensorrt_cfg, network, module_name)
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


def supports_export_stage(export_spec: ExportSpec, stage_name: str) -> bool:
    """Return whether an export specification supports a stage."""
    return stage_name in export_spec.supported_stages
