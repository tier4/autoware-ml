"""Abstract base class and shared export modules for PTv3-based task models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.models.base import BaseModel
from autoware_ml.models.segmentation3d.backbones.ptv3 import (
    PointTransformerV3Backbone,
    SerializedPooling,
    SerializedPoolingMeta,
    build_serialized_pooling_meta,
)
from autoware_ml.ops.indexing.operators import argsort
from autoware_ml.utils.point_cloud.structures import bit_length_tensor, invert_permutation

SERIALIZED_POOLING_FIELDS = tuple(field.name for field in fields(SerializedPoolingMeta))
SERIALIZED_POOLING_INPUT_SIZED_FIELDS = frozenset({"indices", "cluster"})
SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS = frozenset({"indptr"})
SERIALIZED_POOLING_ORDER_FIELDS = frozenset({"serialized_order", "serialized_inverse"})


def validate_serialization_geometry(
    backbone: nn.Module, grid_size: float, point_cloud_range: Sequence[float]
) -> None:
    """Raise if the configured geometry cannot cover the backbone's pooling hierarchy."""
    pooling_depth = sum(
        m.pooling_depth for m in backbone.modules() if isinstance(m, SerializedPooling)
    )
    extent = max(point_cloud_range[i + 3] - point_cloud_range[i] for i in range(3))
    if int(bit_length_tensor(extent / grid_size).item()) < pooling_depth:
        raise ValueError(
            f"point_cloud_range {tuple(point_cloud_range)} with grid_size {grid_size} cannot "
            f"cover the backbone's cumulative pooling depth {pooling_depth}."
        )


class PTv3BaseModel(BaseModel):
    """Abstract base class for all PTv3 task models.

    Provides shared backbone management, export geometry computation, and
    export backbone helpers. Detection and segmentation subclasses inherit
    from this class (potentially with additional base classes via MRO).
    """

    EXPORT_ORDER = ("z", "z-trans")
    EXPORT_SUPPORTED_STAGES = frozenset({"onnx"})

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        grid_size: float | None,
        point_cloud_range: Sequence[float] | None,
        freeze_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 base model.

        Args:
            backbone: PTv3 backbone module.
            grid_size: Voxel grid size used to derive sparse shape and
                serialization depth for export.
            point_cloud_range: Six-element sequence ``[x_min, y_min, z_min,
                x_max, y_max, z_max]`` used to derive sparse shape for export.
            freeze_backbone: When ``True``, the backbone is permanently kept
                in eval mode with its parameters frozen.
            **kwargs: Keyword arguments forwarded to :class:`BaseModel` (and
                further up the MRO chain).
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.grid_size = grid_size
        self.point_cloud_range = (
            tuple(float(v) for v in point_cloud_range) if point_cloud_range is not None else None
        )
        if self.grid_size is not None and self.point_cloud_range is not None:
            validate_serialization_geometry(backbone, self.grid_size, self.point_cloud_range)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> PTv3BaseModel:
        """Keep the frozen backbone in eval mode during training.

        Args:
            mode: When ``True``, set the model to training mode; otherwise to
                evaluation mode.

        Returns:
            This model instance.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Record backbone-freeze provenance in saved checkpoints.

        Args:
            checkpoint: Mutable checkpoint dictionary to annotate.
        """
        checkpoint["autoware_ml_checkpoint_recipe"] = {
            "type": "ptv3",
            "freeze_backbone": self.freeze_backbone,
        }

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int | None:
        """Infer the effective sample batch size for logging.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        if "gt_boxes" in batch_inputs_dict:
            return len(batch_inputs_dict["gt_boxes"])
        if "offset" in batch_inputs_dict:
            return int(batch_inputs_dict["offset"].numel())
        return super().get_log_batch_size(batch_inputs_dict)

    def _compute_export_geometry(
        self, batch_inputs_dict: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse shape and serialization depth for export.

        Args:
            batch_inputs_dict: Preprocessed batch containing at least
                ``coord`` (used for device inference).

        Returns:
            ``(sparse_shape, serialization_depth)`` as long tensors on the
            same device as ``batch_inputs_dict["coord"]``.
        """
        device = batch_inputs_dict["coord"].device
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=torch.float32, device=device)
        axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / self.grid_size
        serialization_depth = bit_length_tensor(torch.max(axis_extents))
        sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
        return sparse_shape, serialization_depth

    def _prepare_backbone_export(self) -> PointTransformerV3Backbone:
        """Return an export-ready copy of the backbone.

        Returns:
            Copy of the backbone prepared for ONNX export with the configured
            export order.
        """
        return self.backbone.prepare_for_export(self.EXPORT_ORDER)


def build_serialized_pooling_metadata(
    grid_coord: torch.Tensor,
    serialized_code: torch.Tensor,
    serialized_order: torch.Tensor,
    strides: Sequence[int],
) -> list[SerializedPoolingMeta]:
    """Build serialized-pooling metadata for every encoder pooling stage."""
    metadata = []
    for stride in strides:
        meta, serialized_code = build_serialized_pooling_meta(
            grid_coord, serialized_code, serialized_order, stride
        )
        metadata.append(meta)
        grid_coord = meta.grid_coord
        serialized_order = meta.serialized_order
    return metadata


def flatten_serialized_pooling_inputs(
    metadata: Sequence[SerializedPoolingMeta],
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Flatten per-stage metadata into ONNX args and input names."""
    inputs: list[torch.Tensor] = []
    names: list[str] = []
    for stage_index, meta in enumerate(metadata):
        for field in SERIALIZED_POOLING_FIELDS:
            inputs.append(getattr(meta, field))
            names.append(f"serialized_pooling_{stage_index}_{field}")
    return tuple(inputs), names


def _serialized_pooling_dynamic_axis(input_name: str) -> dict[int, str]:
    _, _, stage_index, field = input_name.split("_", 3)
    stage_prefix = f"serialized_pooling_{stage_index}"
    if field in SERIALIZED_POOLING_INPUT_SIZED_FIELDS:
        return {0: f"{stage_prefix}_in_voxels"}
    if field in SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS:
        return {0: f"{stage_prefix}_out_voxels_plus_one"}
    if field in SERIALIZED_POOLING_ORDER_FIELDS:
        return {1: f"{stage_prefix}_out_voxels"}
    return {0: f"{stage_prefix}_out_voxels"}


def build_point_feature_dynamic_axes(tensor_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for tensors indexed by the decoded point/voxel count."""
    return {tensor_name: {0: "num_voxels"} for tensor_name in tensor_names}


def build_ptv3_input_dynamic_axes(input_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for generated PTv3 backbone export inputs."""
    dynamic_axes: dict[str, dict[int, str]] = {}
    for input_name in input_names:
        if input_name in {"grid_coord", "feat"}:
            dynamic_axes[input_name] = {0: "num_voxels"}
        elif input_name == "serialized_code":
            dynamic_axes[input_name] = {1: "num_voxels"}
        elif input_name.startswith("serialized_pooling_"):
            dynamic_axes[input_name] = _serialized_pooling_dynamic_axis(input_name)
    return dynamic_axes


def build_ptv3_backbone_dynamic_axes(input_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split PTv3 backbone export graph."""
    dynamic_axes = build_ptv3_input_dynamic_axes(input_names)
    dynamic_axes.update(build_point_feature_dynamic_axes(("point_feat", "point_grid_coord")))
    return dynamic_axes


def build_serialized_pooling_export_inputs(
    grid_coord: torch.Tensor,
    serialized_code: torch.Tensor,
    serialized_order: torch.Tensor,
    strides: Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Build flattened serialized-pooling sample tensors and their ONNX input names."""
    return flatten_serialized_pooling_inputs(
        build_serialized_pooling_metadata(grid_coord, serialized_code, serialized_order, strides)
    )


def make_serialized_pooling_from_flat_inputs(
    serialized_pooling_inputs: tuple[torch.Tensor, ...],
) -> list[SerializedPoolingMeta]:
    """Reconstruct per-stage metadata objects from flattened ONNX graph inputs."""
    num_fields = len(SERIALIZED_POOLING_FIELDS)
    if len(serialized_pooling_inputs) % num_fields != 0:
        raise ValueError("serialized-pooling inputs are not divisible by metadata field count.")
    return [
        SerializedPoolingMeta(*serialized_pooling_inputs[index : index + num_fields])
        for index in range(0, len(serialized_pooling_inputs), num_fields)
    ]


def _run_ptv3_backbone_export(
    backbone: PointTransformerV3Backbone,
    grid_coord: torch.Tensor,
    feat: torch.Tensor,
    serialized_depth: torch.Tensor,
    serialized_code: torch.Tensor,
    sparse_shape: torch.Tensor,
    *serialized_pooling_inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the shared tensor-only PTv3 backbone export path."""
    point_count = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
    serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
    serialized_inverse = invert_permutation(serialized_order)
    point = backbone.export_forward(
        {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": grid_coord,
            "offset": point_count,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "serialized_pooling": make_serialized_pooling_from_flat_inputs(
                serialized_pooling_inputs
            ),
            "sparse_shape": sparse_shape,
        }
    )
    return point.feat, point.grid_coord, point.offset


class _PTv3BackboneExportModule(nn.Module):
    """Export-only PTv3 backbone producing raw point features."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
    ) -> None:
        """Initialize the backbone export module.

        Args:
            backbone: Export-prepared PTv3 backbone copy.
            sparse_shape: Static sparse shape baked at export time.
            serialized_depth: Serialization depth baked at export time.
        """
        super().__init__()
        self.backbone = backbone
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run backbone and return raw point features with coordinates and batch offsets."""
        return _run_ptv3_backbone_export(
            self.backbone,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )


class _PTv3SegHeadExportModule(nn.Module):
    """Export-only segmentation head consuming backbone point features."""

    def __init__(self, seg3d_head: nn.Module) -> None:
        """Initialize the segmentation head export module.

        Args:
            seg3d_head: Segmentation head copy.
        """
        super().__init__()
        self.seg3d_head = seg3d_head

    def forward(self, point_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply segmentation head and return label predictions and class probabilities."""
        logits = self.seg3d_head(point_feat)
        probs = torch.softmax(logits, dim=1)
        return probs.argmax(dim=1), probs
