"""PTv3-based lidar detection models.

This module adapts the point-based PTv3 backbone to dense BEV detection heads
by projecting decoded point features onto a trainable BEV canvas and refining
them with an explicit BEV encoder.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.metrics.detection3d import CenterDistanceMeanAP
from autoware_ml.models.detection3d.base import Detection3DBaseModel
from autoware_ml.models.segmentation3d.backbones.ptv3 import PointTransformerV3Backbone
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    _PTv3BackboneExportModule,
    _run_ptv3_backbone_export,
    build_point_feature_dynamic_axes,
    build_ptv3_backbone_dynamic_axes,
    build_ptv3_input_dynamic_axes,
    build_serialized_pooling_export_inputs,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import offset_to_batch
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch


class PTv3BEVProjection(nn.Module):
    """Project decoded PTv3 point features onto a dense BEV canvas."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: Sequence[int],
        bev_stride: int = 1,
    ) -> None:
        """Initialize the PTv3-to-BEV projection path.

        Args:
            in_channels: PTv3 point-feature dimension.
            out_channels: Dense BEV channel dimension.
            output_shape: Dense BEV shape as ``(height, width)``.
            bev_stride: Integer downsampling factor applied to XY grid cells
                before scattering them into the dense BEV canvas.
        """
        super().__init__()
        self.output_shape = tuple(int(value) for value in output_shape)
        self.out_channels = out_channels
        self.bev_stride = int(bev_stride)
        self.point_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.bev_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter PTv3 point features into a dense BEV tensor.

        Args:
            point_features: Decoded PTv3 point features.
            grid_coord: Absolute voxel-grid coordinates in ``(x, y, z)`` order.
            offset: Cumulative point offsets for each batch element.

        Returns:
            Dense BEV feature tensor with shape ``(B, C, H, W)``.
        """
        batch_size = int(offset.numel())
        height, width = self.output_shape
        if point_features.numel() == 0:
            return point_features.new_zeros((batch_size, self.out_channels, height, width))

        projected_features = self.point_proj(point_features)
        batch_indices = offset_to_batch(offset, grid_coord)
        x_indices = torch.div(grid_coord[:, 0].long(), self.bev_stride, rounding_mode="floor")
        y_indices = torch.div(grid_coord[:, 1].long(), self.bev_stride, rounding_mode="floor")
        valid_mask = (
            (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
        )
        if not torch.any(valid_mask):
            return projected_features.new_zeros((batch_size, self.out_channels, height, width))

        projected_features = projected_features[valid_mask]
        batch_indices = batch_indices[valid_mask]
        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]

        flat_indices = batch_indices * (height * width) + y_indices * width + x_indices
        canvas = projected_features.new_zeros((batch_size * height * width, self.out_channels))
        scatter_indices = flat_indices.unsqueeze(1).expand(-1, self.out_channels)
        canvas.scatter_reduce_(
            dim=0,
            index=scatter_indices,
            src=projected_features,
            reduce="amax",
            include_self=True,
        )
        bev = canvas.view(batch_size, height, width, self.out_channels)
        bev = bev.permute(0, 3, 1, 2).contiguous()
        return self.bev_refine(bev)


class PTv3BEVResidualBlock(nn.Module):
    """Refine dense BEV features with a residual 2D convolution block."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        """Initialize the residual BEV block.

        Args:
            in_channels: Input BEV channel dimension.
            out_channels: Output BEV channel dimension.
            dilation: Dilation applied to the 3x3 convolutions.
        """
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one residual refinement step."""
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class PTv3BEVEncoder(nn.Module):
    """Refine scattered PTv3 BEV features before dense detection heads."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dilations: Sequence[int],
    ) -> None:
        """Initialize the PTv3 BEV encoder.

        Args:
            in_channels: Input channel count from the projector.
            hidden_channels: Internal channel width for intermediate blocks.
            out_channels: Final channel count consumed by the detection head.
            dilations: Dilation schedule for the residual refinement blocks.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        current_channels = hidden_channels
        for block_index, dilation in enumerate(dilations):
            next_channels = out_channels if block_index == len(dilations) - 1 else hidden_channels
            blocks.append(PTv3BEVResidualBlock(current_channels, next_channels, dilation=dilation))
            current_channels = next_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BEV features aligned to the configured detection head."""
        return self.blocks(self.stem(x))


class _PTv3DetectionExportModule(nn.Module):
    """Export PTv3 detection as a tensor-only ONNX graph."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        bev_projector: PTv3BEVProjection,
        bev_encoder: nn.Module,
        bbox_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        output_names: Sequence[str],
    ) -> None:
        """Initialize the deployment-oriented PTv3 detection wrapper."""
        super().__init__()
        self.backbone = backbone
        self.bev_projector = bev_projector
        self.bev_encoder = bev_encoder
        self.bbox_head = bbox_head
        self.output_names = list(output_names)
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run export-time inference on serialized point inputs."""
        point_feat, point_grid_coord, point_offset = _run_ptv3_backbone_export(
            self.backbone,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )
        bev_features = self.bev_projector(point_feat, point_grid_coord, point_offset)
        bev_features = self.bev_encoder(bev_features)
        outputs = self.bbox_head(bev_features)
        return tuple(outputs[name] for name in self.output_names)


class _PTv3DetHeadExportModule(nn.Module):
    """Export-only detection head consuming backbone point features."""

    def __init__(
        self,
        bev_projector: PTv3BEVProjection,
        bev_encoder: nn.Module,
        bbox_head: nn.Module,
        output_names: Sequence[str],
    ) -> None:
        """Initialize the export-only detection head module.

        Args:
            bev_projector: Export-ready point-to-BEV projection module.
            bev_encoder: Export-ready BEV encoder module.
            bbox_head: Export-ready detection head module.
            output_names: Ordered output tensor names emitted by ``bbox_head``.
        """
        super().__init__()
        self.bev_projector = bev_projector
        self.bev_encoder = bev_encoder
        self.bbox_head = bbox_head
        self.output_names = list(output_names)

    def forward(
        self,
        point_feat: torch.Tensor,
        point_grid_coord: torch.Tensor,
        point_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Project point features to BEV and run the detection head."""
        bev = self.bev_projector(point_feat, point_grid_coord, point_offset)
        bev = self.bev_encoder(bev)
        outputs = self.bbox_head(bev)
        return tuple(outputs[name] for name in self.output_names)


class PTv3DetectionModel(PTv3BaseModel, Detection3DBaseModel):
    """Compose PTv3 features with dense BEV detection heads."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        bev_projector: PTv3BEVProjection,
        bev_encoder: nn.Module,
        bbox_head: nn.Module,
        export_output_names: Sequence[str],
        grid_size: float,
        point_cloud_range: Sequence[float],
        freeze_backbone: bool = False,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metric: CenterDistanceMeanAP | None = None,
    ) -> None:
        """Initialize the PTv3 detection model."""
        super().__init__(
            backbone=backbone,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            freeze_backbone=freeze_backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            metric=metric,
        )
        self.bev_projector = bev_projector
        self.bev_encoder = bev_encoder
        self.bbox_head = bbox_head
        self.export_output_names = list(export_output_names)

    def get_export_output_names(self) -> list[str]:
        """Return the ordered export output names."""
        return list(self.export_output_names)

    def _extract_bev_features(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Encode PTv3 point features and project them into BEV."""
        point = self.backbone(
            {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}
        )
        bev_features = self.bev_projector(point.feat, point.grid_coord, point.offset)
        return self.bev_encoder(bev_features)

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run PTv3 feature extraction followed by the configured detection head."""
        bev_features = self._extract_bev_features(coord, feat, grid_coord, offset)
        return self.bbox_head(bev_features)

    def compute_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute detection losses for one batched step."""
        return self.bbox_head.loss(
            outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )

    def predict_outputs(
        self, batch_inputs_dict: Mapping[str, Any], outputs: dict[str, torch.Tensor]
    ) -> Any:
        """Decode predictions for inference."""
        del batch_inputs_dict
        return self.bbox_head.predict(outputs)

    def build_export_spec(self, batch_inputs_dict: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the PTv3 detection ONNX export specification."""
        sparse_shape, serialization_depth = self._compute_export_geometry(batch_inputs_dict)
        point, input_args = serialize_point_cloud_batch(
            batch_inputs_dict, self.EXPORT_ORDER, serialization_depth
        )
        serialized_pooling_inputs, serialized_pooling_input_names = (
            build_serialized_pooling_export_inputs(
                point["grid_coord"],
                point["serialized_code"],
                point["serialized_order"],
                self.backbone.stride,
            )
        )
        export_module = _PTv3DetectionExportModule(
            backbone=self._prepare_backbone_export(),
            bev_projector=deepcopy(self.bev_projector).eval(),
            bev_encoder=deepcopy(self.bev_encoder).eval(),
            bbox_head=self.bbox_head.prepare_for_export(),
            sparse_shape=sparse_shape,
            serialized_depth=serialization_depth,
            output_names=self.export_output_names,
        )
        export_module.eval()
        export_input_args = (
            input_args[0],
            input_args[1],
            input_args[3],
            *serialized_pooling_inputs,
        )
        input_param_names = [
            "grid_coord",
            "feat",
            "serialized_code",
            *serialized_pooling_input_names,
        ]
        return ExportSpec(
            module=export_module,
            args=export_input_args,
            input_param_names=input_param_names,
            output_names=self.get_export_output_names(),
            dynamic_axes=build_ptv3_input_dynamic_axes(input_param_names),
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(
        self, batch_inputs_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, ExportSpec]:
        """Build split PTv3 detection ONNX export specs for backbone and detection head."""
        sparse_shape, serialization_depth = self._compute_export_geometry(batch_inputs_dict)
        point, input_args = serialize_point_cloud_batch(
            batch_inputs_dict, self.EXPORT_ORDER, serialization_depth
        )
        serialized_pooling_inputs, serialized_pooling_input_names = (
            build_serialized_pooling_export_inputs(
                point["grid_coord"],
                point["serialized_code"],
                point["serialized_order"],
                self.backbone.stride,
            )
        )
        backbone_input_args = (
            input_args[0],
            input_args[1],
            input_args[3],
            *serialized_pooling_inputs,
        )
        backbone_input_names = [
            "grid_coord",
            "feat",
            "serialized_code",
            *serialized_pooling_input_names,
        ]

        backbone_module = _PTv3BackboneExportModule(
            backbone=self._prepare_backbone_export(),
            sparse_shape=sparse_shape,
            serialized_depth=serialization_depth,
        ).eval()
        with torch.no_grad():
            point_feat, point_grid_coord, point_offset = backbone_module(*backbone_input_args)

        det3d_head_module = _PTv3DetHeadExportModule(
            bev_projector=deepcopy(self.bev_projector).eval(),
            bev_encoder=deepcopy(self.bev_encoder).eval(),
            bbox_head=self.bbox_head.prepare_for_export(),
            output_names=self.export_output_names,
        ).eval()

        return {
            "backbone": ExportSpec(
                module=backbone_module,
                args=backbone_input_args,
                input_param_names=backbone_input_names,
                output_names=["point_feat", "point_grid_coord", "point_offset"],
                dynamic_axes=build_ptv3_backbone_dynamic_axes(backbone_input_names),
                supported_stages=self.EXPORT_SUPPORTED_STAGES,
            ),
            "det3d_head": ExportSpec(
                module=det3d_head_module,
                args=(point_feat, point_grid_coord, point_offset),
                input_param_names=["point_feat", "point_grid_coord", "point_offset"],
                output_names=self.export_output_names,
                dynamic_axes=build_point_feature_dynamic_axes(("point_feat", "point_grid_coord")),
                supported_stages=self.EXPORT_SUPPORTED_STAGES,
            ),
        }
