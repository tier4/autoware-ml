"""PTv3-based lidar detection models.

This module adapts the PTv3 encoder to dense BEV detection heads. The
detection branch taps the two coarsest encoder stages directly: a feature
fusion mirrors one decoder unpooling step to bring global context to the BEV
resolution, and the fused features are scattered onto a trainable BEV canvas
and refined with an explicit BEV encoder. The PTv3 decoder is owned by the
segmentation head and is not part of the detection path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    _run_ptv3_encoder_export,
    build_encoder_export_spec,
    build_ptv3_export_context,
    build_ptv3_input_dynamic_axes,
    build_serialized_pooling_export_inputs,
    stage_voxel_axis_name,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import offset_to_batch
from autoware_ml.utils.point_cloud.structures import Point, serialize_point_cloud_batch


class PTv3DetFeatureFusion(nn.Module):
    """Fuse the deepest encoder stage into the BEV-resolution encoder stage.

    This mirrors one decoder unpooling step (coarse projection gathered
    through the pooling inverse plus a skip projection), but reads the
    ``pooling_parent``/``pooling_inverse`` chain non-destructively: nothing is
    popped and no parent features are mutated, so the segmentation decoder can
    still consume the intact chain afterwards.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialize the detection feature fusion.

        Args:
            in_channels: Deepest encoder stage feature dimension.
            skip_channels: BEV-resolution encoder stage feature dimension.
            out_channels: Fused feature dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )

    def forward(self, point: Point) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse the deepest stage into its parent stage.

        Args:
            point: Deepest encoder point with its pooling chain attached.

        Returns:
            Tuple of fused features, parent-stage voxel coordinates, and
            parent-stage batch offsets.
        """
        parent = point["pooling_parent"]
        inverse = point["pooling_inverse"]
        fused = self.proj_skip(parent.feat) + self.proj(point.feat)[inverse]
        return fused, parent.grid_coord, parent.offset


class PTv3BEVProjection(nn.Module):
    """Project PTv3 point features onto a dense BEV canvas."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: Sequence[int],
    ) -> None:
        """Initialize the PTv3-to-BEV projection path.

        Args:
            in_channels: PTv3 point-feature dimension.
            out_channels: Dense BEV channel dimension.
            output_shape: Dense BEV shape as ``(height, width)``.
        """
        super().__init__()
        self.output_shape = tuple(int(value) for value in output_shape)
        self.out_channels = out_channels
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
            point_features: PTv3 point features.
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
        x_indices = grid_coord[:, 0].long()
        y_indices = grid_coord[:, 1].long()
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


class PTv3DetBEVNeck(nn.Module):
    """Turn the encoder pooling chain into dense BEV features for detection.

    Composes the detection feature fusion, BEV projection, and BEV encoder so
    the detection-only and joint segdet models share one detection branch
    implementation and state-dict layout.
    """

    def __init__(
        self,
        fusion: PTv3DetFeatureFusion,
        bev_projector: PTv3BEVProjection,
        bev_encoder: PTv3BEVEncoder,
    ) -> None:
        """Initialize the detection BEV neck.

        Args:
            fusion: Encoder-stage fusion producing BEV-resolution features.
            bev_projector: Projects fused features into a dense BEV grid.
            bev_encoder: Dense BEV feature encoder.
        """
        super().__init__()
        self.fusion = fusion
        self.bev_projector = bev_projector
        self.bev_encoder = bev_encoder

    def forward(self, point: Point) -> torch.Tensor:
        """Produce dense BEV features from the deepest encoder point.

        Args:
            point: Deepest encoder point with its pooling chain attached.

        Returns:
            Dense BEV feature tensor with shape ``(B, C, H, W)``.
        """
        fused, grid_coord, offset = self.fusion(point)
        bev = self.bev_projector(fused, grid_coord, offset)
        return self.bev_encoder(bev)


def det_head_export_input_names(stage_count: int) -> list[str]:
    """Return the split det-head export input names for a given stage count."""
    skip_stage = stage_count - 2
    deep_stage = stage_count - 1
    return [
        f"point_feat_{skip_stage}",
        f"point_feat_{deep_stage}",
        f"pooling_cluster_{skip_stage}",
        f"point_grid_coord_{skip_stage}",
    ]


def det_head_export_dynamic_axes(stage_count: int) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split det-head export graph inputs."""
    skip_stage = stage_count - 2
    deep_stage = stage_count - 1
    return {
        f"point_feat_{skip_stage}": {0: stage_voxel_axis_name(skip_stage)},
        f"point_feat_{deep_stage}": {0: stage_voxel_axis_name(deep_stage)},
        f"pooling_cluster_{skip_stage}": {0: f"serialized_pooling_{skip_stage}_in_voxels"},
        f"point_grid_coord_{skip_stage}": {0: stage_voxel_axis_name(skip_stage)},
    }


def build_det_head_export_spec(
    context: "PTv3ExportContext",
    bev_neck: PTv3DetBEVNeck,
    bbox_head: nn.Module,
    output_names: Sequence[str],
) -> ExportSpec:
    """Build the detection-head export spec from the two coarsest encoder stages.

    Args:
        context: Shared export context.
        bev_neck: Detection BEV neck (copied internally).
        bbox_head: Export-prepared detection head copy.
        output_names: Ordered head output names.
    """
    module = _PTv3DetHeadExportModule(
        bev_neck=deepcopy(bev_neck).eval(),
        bbox_head=bbox_head,
        output_names=output_names,
    ).eval()
    skip_stage = context.stage_count - 2
    skip_grid_coord = (
        context.grid_coord
        if skip_stage == 0
        else context.pooling_metadata[skip_stage - 1].grid_coord
    )
    return ExportSpec(
        module=module,
        args=(
            context.stage_feats[skip_stage],
            context.stage_feats[context.stage_count - 1],
            context.pooling_metadata[skip_stage].cluster,
            skip_grid_coord,
        ),
        input_param_names=det_head_export_input_names(context.stage_count),
        output_names=list(output_names),
        dynamic_axes=det_head_export_dynamic_axes(context.stage_count),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


class _PTv3DetectionExportModule(nn.Module):
    """Export PTv3 detection as a tensor-only ONNX graph."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        output_names: Sequence[str],
    ) -> None:
        """Initialize the deployment-oriented PTv3 detection wrapper."""
        super().__init__()
        self.encoder = encoder
        self.bev_neck = bev_neck
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
        point = _run_ptv3_encoder_export(
            self.encoder,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )
        bev_features = self.bev_neck(point)
        outputs = self.bbox_head(bev_features)
        return tuple(outputs[name] for name in self.output_names)


class _PTv3DetHeadExportModule(nn.Module):
    """Export-only detection head consuming the two coarsest encoder stages."""

    def __init__(
        self,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        output_names: Sequence[str],
    ) -> None:
        """Initialize the export-only detection head module.

        Args:
            bev_neck: Export-ready detection BEV neck.
            bbox_head: Export-ready detection head module.
            output_names: Ordered output tensor names emitted by ``bbox_head``.
        """
        super().__init__()
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.output_names = list(output_names)

    def forward(
        self,
        skip_feat: torch.Tensor,
        deepest_feat: torch.Tensor,
        cluster: torch.Tensor,
        skip_grid_coord: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Rebuild the coarse pooling link, project to BEV, and run the head."""
        offset = shape_as_tensor(skip_feat)[:1].to(skip_feat.device)
        parent = Point(feat=skip_feat, grid_coord=skip_grid_coord, offset=offset)
        point = Point(feat=deepest_feat, pooling_parent=parent, pooling_inverse=cluster)
        bev = self.bev_neck(point)
        outputs = self.bbox_head(bev)
        return tuple(outputs[name] for name in self.output_names)


class PTv3DetectionModel(PTv3BaseModel):
    """Compose the PTv3 encoder with dense BEV detection heads."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        export_output_names: Sequence[str],
        grid_size: float,
        point_cloud_range: Sequence[float],
        freeze_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 detection model.

        Args:
            encoder: PTv3 encoder module.
            bev_neck: Detection BEV neck consuming the encoder pooling chain.
            bbox_head: Detection head producing the decoded predictions.
            export_output_names: Ordered output names used during export.
            grid_size: Voxel grid size used to derive sparse shape.
            point_cloud_range: Point-cloud range used to derive sparse shape.
            freeze_encoder: When ``True``, keep the encoder frozen in eval mode.
            **kwargs: Keyword arguments forwarded to :class:`BaseModel`.
        """
        super().__init__(
            encoder=encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.export_output_names = list(export_output_names)

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.bbox_head.predict(outputs), batch)

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
        point = self.encoder(
            {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}
        )
        return self.bev_neck(point)

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
                self.encoder.stride,
            )
        )
        export_module = _PTv3DetectionExportModule(
            encoder=self._prepare_encoder_export(),
            bev_neck=deepcopy(self.bev_neck).eval(),
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
        """Build split PTv3 detection ONNX export specs for encoder and detection head."""
        context = build_ptv3_export_context(self, batch_inputs_dict)
        return {
            "encoder": build_encoder_export_spec(context),
            "det3d_head": build_det_head_export_spec(
                context,
                self.bev_neck,
                self.bbox_head.prepare_for_export(),
                self.export_output_names,
            ),
        }
