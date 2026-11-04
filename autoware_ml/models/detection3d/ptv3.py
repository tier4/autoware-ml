"""PTv3-based lidar detection models.

This module adapts the PTv3 encoder to dense BEV detection heads. The
detection branch taps the two coarsest encoder stages directly: a feature
fusion mirrors one decoder unpooling step to bring global context to the BEV
resolution, and the fused features are scattered onto a trainable BEV canvas
and refined with an explicit BEV encoder. The PTv3 decoder is owned by the
segmentation head and is not part of the detection path.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.segmentation3d.ptv3_base import (
    SERIALIZED_POOLING_FIELDS,
    PTv3BaseModel,
    PTv3EncoderExportBase,
    PTv3ExportContext,
    build_encoder_export_spec,
    build_ptv3_export_context,
    build_ptv3_input_dynamic_axes,
    prepare_ptv3_export_inputs,
    stage_voxel_axis_name,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import offset_to_batch
from autoware_ml.utils.point_cloud.structures import Point


class PTv3Det3DFeatureFusion(nn.Module):
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


def _linear_bn_gelu(in_channels: int, out_channels: int) -> nn.Sequential:
    """Return a point-wise projection with batch norm and GELU."""
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.GELU(),
    )


class PTv3Det3DFeaturePyramid(nn.Module):
    """Fuse the deepest encoder stage top-down into the finer stages and keep every fused stage.

    Every step mirrors one decoder unpooling step, reading the pooling chain without changing it.
    With ``keep_deepest`` the projected deepest stage is a level too, the coarsest one.
    """

    def __init__(
        self, stage_channels: Sequence[int], out_channels: int, keep_deepest: bool = False
    ) -> None:
        """Initialize the feature pyramid.

        Args:
            stage_channels: Feature dimensions of the stages from the finest fused one to the
                deepest one.
            out_channels: Fused feature dimension.
            keep_deepest: Whether the projected deepest stage is returned as a level.

        Raises:
            ValueError: If fewer than two stages are given.
        """
        super().__init__()
        if len(stage_channels) < 2:
            raise ValueError(f"The pyramid fuses at least two stages, got {list(stage_channels)}.")
        self.keep_deepest = keep_deepest
        self.proj_skip = nn.ModuleList(
            _linear_bn_gelu(channels, out_channels) for channels in stage_channels[:-1]
        )
        # The deepest stage feeds the first fusion, every fused stage the next finer one
        self.proj_coarse = nn.ModuleList(
            [
                _linear_bn_gelu(stage_channels[-1], out_channels),
                *(_linear_bn_gelu(out_channels, out_channels) for _ in stage_channels[:-2]),
            ]
        )

    @property
    def num_stages(self) -> int:
        """Number of encoder stages the pyramid reads."""
        return len(self.proj_skip) + 1

    @property
    def num_levels(self) -> int:
        """Number of returned levels."""
        return len(self.proj_skip) + int(self.keep_deepest)

    def forward(self, point: Point) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fuse the deepest stage down to the finest fused stage.

        Args:
            point: Deepest encoder point with its pooling chain attached.

        Returns:
            Features, voxel coordinates and batch offsets of every level, from the finest to the
            coarsest.
        """
        deepest = self.proj_coarse[0](point.feat)
        levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        child, coarse = point, deepest
        # The fused stage of every step is projected for the next finer one, the last is not used
        next_projections = (*self.proj_coarse[1:], nn.Identity())
        for proj_skip, proj_next in zip(reversed(self.proj_skip), next_projections, strict=True):
            parent = child["pooling_parent"]
            fused = proj_skip(parent.feat) + coarse[child["pooling_inverse"]]
            levels.append((fused, parent.grid_coord, parent.offset))
            child, coarse = parent, proj_next(fused)
        deepest_level = [(deepest, point.grid_coord, point.offset)] if self.keep_deepest else []
        return levels[::-1] + deepest_level


class PTv3BEVProjection(nn.Module):
    """Project PTv3 point features onto a dense BEV canvas."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: Sequence[int],
        assume_valid_grid_coord: bool = False,
    ) -> None:
        """Initialize the PTv3-to-BEV projection path.

        Args:
            in_channels: PTv3 point-feature dimension.
            out_channels: Dense BEV channel dimension.
            output_shape: Dense BEV shape as ``(height, width)``.
            assume_valid_grid_coord: Skip BEV bounds filtering. Use only when
                preprocessing guarantees all ``grid_coord`` values are within
                ``output_shape``; this avoids data-dependent ``NonZero`` nodes
                in exported ONNX graphs.
        """
        super().__init__()
        self.output_shape = tuple(int(value) for value in output_shape)
        self.out_channels = out_channels
        self.assume_valid_grid_coord = assume_valid_grid_coord
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
        if not self.assume_valid_grid_coord:
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


class PTv3Det3DBEVNeck(nn.Module):
    """Turn the encoder pooling chain into dense BEV features for detection.

    Composes the detection feature fusion, BEV projection, and BEV encoder so
    the detection-only and joint det3d_seg3d models share one detection branch
    implementation and state-dict layout.
    """

    def __init__(
        self,
        fusion: PTv3Det3DFeatureFusion,
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


class PTv3StageBEVPyramid(nn.Module):
    """Dense BEV maps of fused sparse stages for heads that sample them at (x, y).

    Every projected stage scatters its voxels onto the BEV grid of its stride with the maximum
    over height. The coarsest projected map runs through a BEV encoder and halves into the
    downsampled levels.
    """

    def __init__(
        self,
        *,
        point_cloud_range: Sequence[float],
        grid_size: float,
        stage_strides: Sequence[int],
        projected_strides: Sequence[int],
        in_channels: int,
        channels: int,
        bev_encoder: PTv3BEVEncoder,
        num_downsampled: int = 0,
    ) -> None:
        """Initialize the BEV pyramid.

        Args:
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            stage_strides: Total pooling strides of the fused stages, finest first.
            projected_strides: Strides of the stages projected onto BEV, finest first.
            in_channels: Feature dimension of the fused stages.
            channels: Feature dimension of the BEV maps.
            bev_encoder: Encoder of the coarsest projected map.
            num_downsampled: Levels halved from the encoded map, coarser than every stage.

        Raises:
            ValueError: If a projected stride is not a stage stride or the strides are unsorted.
        """
        super().__init__()
        missing = [stride for stride in projected_strides if stride not in stage_strides]
        if missing or list(projected_strides) != sorted(projected_strides):
            raise ValueError(
                f"projected_strides {list(projected_strides)} must be sorted stage strides "
                f"{list(stage_strides)}."
            )
        cells = [
            round((point_cloud_range[axis + 3] - point_cloud_range[axis]) / grid_size)
            for axis in (0, 1)
        ]
        self.stage_indices = [list(stage_strides).index(stride) for stride in projected_strides]
        self.grid_origin = (float(point_cloud_range[0]), float(point_cloud_range[1]))
        self.cell_sizes = [grid_size * stride for stride in projected_strides] + [
            grid_size * projected_strides[-1] * 2 ** (level + 1) for level in range(num_downsampled)
        ]
        self.projections = nn.ModuleList(
            PTv3BEVProjection(
                in_channels,
                channels,
                output_shape=(-(-cells[1] // stride), -(-cells[0] // stride)),
                assume_valid_grid_coord=True,
            )
            for stride in projected_strides
        )
        self.bev_encoder = bev_encoder
        self.downsample = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_downsampled)
        )

    @property
    def num_levels(self) -> int:
        """Number of returned BEV levels."""
        return len(self.cell_sizes)

    def forward(
        self, stages: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> list[BEVLevel]:
        """Project the stages onto BEV and build the coarser levels.

        Args:
            stages: Features, voxel coordinates and batch offsets of the fused stages, finest
                first.

        Returns:
            BEV levels from the finest to the coarsest.
        """
        maps = [
            projection(*stages[index])
            for projection, index in zip(self.projections, self.stage_indices, strict=True)
        ]
        maps[-1] = self.bev_encoder(maps[-1])
        for downsample in self.downsample:
            maps.append(downsample(maps[-1]))
        return [
            BEVLevel(feat, self.grid_origin, cell_size)
            for feat, cell_size in zip(maps, self.cell_sizes, strict=True)
        ]


def _grid_coord_stages(stages: range, deepest_grid_coord: bool) -> range:
    """Stages whose voxel coordinates the det head reads, the deepest one only when it is sampled."""
    return stages if deepest_grid_coord else stages[:-1]


def det3d_head_export_input_names(
    stage_count: int, finest_stage: int, deepest_grid_coord: bool = False
) -> list[str]:
    """Return the split det-head export input names of the stages from ``finest_stage`` on."""
    stages = range(finest_stage, stage_count)
    return [
        *(f"point_feat_{stage}" for stage in stages),
        *(f"pooling_cluster_{stage}" for stage in stages[:-1]),
        *(f"point_grid_coord_{stage}" for stage in _grid_coord_stages(stages, deepest_grid_coord)),
    ]


def det3d_head_export_dynamic_axes(
    stage_count: int, finest_stage: int, deepest_grid_coord: bool = False
) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split det-head export inputs of the stages from ``finest_stage``."""
    stages = range(finest_stage, stage_count)
    return {
        **{f"point_feat_{stage}": {0: stage_voxel_axis_name(stage)} for stage in stages},
        **{
            f"pooling_cluster_{stage}": {0: f"serialized_pooling_{stage}_in_voxels"}
            for stage in stages[:-1]
        },
        **{
            f"point_grid_coord_{stage}": {0: stage_voxel_axis_name(stage)}
            for stage in _grid_coord_stages(stages, deepest_grid_coord)
        },
    }


def det3d_head_export_args(
    context: "PTv3ExportContext", finest_stage: int, deepest_grid_coord: bool = False
) -> tuple[torch.Tensor, ...]:
    """Return the split det-head export inputs of the encoder stages from ``finest_stage`` on."""
    stages = range(finest_stage, context.stage_count)
    grid_coords = (
        context.inputs.grid_coord,
        *(metadata.grid_coord for metadata in context.inputs.pooling_metadata),
    )
    return (
        *(context.stage_feats[stage] for stage in stages),
        *(context.inputs.pooling_metadata[stage].cluster for stage in stages[:-1]),
        *(grid_coords[stage] for stage in _grid_coord_stages(stages, deepest_grid_coord)),
    )


def rebuild_point_chain(
    feats: Sequence[torch.Tensor],
    clusters: Sequence[torch.Tensor],
    grid_coords: Sequence[torch.Tensor],
) -> Point:
    """Rebuild the deepest point and its pooling chain from the split det-head export inputs.

    Args:
        feats: Features of the stages from the finest to the deepest one.
        clusters: Pooling clusters of every stage but the deepest.
        grid_coords: Voxel coordinates of every stage but the deepest, or of every stage when the
            deepest one is sampled too.

    Returns:
        Deepest point with the rebuilt pooling chain.

    Raises:
        ValueError: If the voxel coordinates match neither layout.
    """
    if len(grid_coords) not in (len(feats) - 1, len(feats)):
        raise ValueError(
            f"Expected voxel coordinates of {len(feats) - 1} or {len(feats)} stages, "
            f"got {len(grid_coords)}."
        )
    point = Point(feat=feats[0], grid_coord=grid_coords[0], offset=_single_offset(feats[0]))
    for feat, grid_coord, cluster in zip(
        feats[1:-1], grid_coords[1 : len(feats) - 1], clusters[:-1], strict=True
    ):
        point = Point(
            feat=feat,
            grid_coord=grid_coord,
            offset=_single_offset(feat),
            pooling_parent=point,
            pooling_inverse=cluster,
        )
    sampled_deepest = (
        {"grid_coord": grid_coords[-1], "offset": _single_offset(feats[-1])}
        if len(grid_coords) == len(feats)
        else {}
    )
    return Point(
        feat=feats[-1], pooling_parent=point, pooling_inverse=clusters[-1], **sampled_deepest
    )


def _single_offset(feat: torch.Tensor) -> torch.Tensor:
    """Batch offsets of a single sample export graph."""
    return shape_as_tensor(feat)[:1].to(feat.device)


def build_det3d_head_export_spec(
    context: "PTv3ExportContext",
    bev_neck: PTv3Det3DBEVNeck,
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
    module = _PTv3Det3DHeadExportModule(
        bev_neck=deepcopy(bev_neck).eval(),
        bbox_head=bbox_head,
        output_names=output_names,
    ).eval()
    skip_stage = context.stage_count - 2
    return ExportSpec(
        module=module,
        args=det3d_head_export_args(context, skip_stage),
        input_names=det3d_head_export_input_names(context.stage_count, skip_stage),
        output_names=list(output_names),
        dynamic_axes=det3d_head_export_dynamic_axes(context.stage_count, skip_stage),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


class _PTv3Detection3DExportModule(PTv3EncoderExportBase):
    """Export PTv3 detection as a tensor-only ONNX graph."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        bev_neck: PTv3Det3DBEVNeck,
        bbox_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        output_names: Sequence[str],
    ) -> None:
        """Initialize the deployment-oriented PTv3 detection wrapper."""
        super().__init__(encoder, voxel_encoder, sparse_shape, serialized_depth)
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.output_names = list(output_names)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        grid_coord: torch.Tensor,
        patch_order: torch.Tensor,
        serialized_inverse: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run export-time inference on serialized voxel inputs."""
        point = self.run_encoder(
            voxels,
            num_points_per_voxel,
            grid_coord,
            patch_order,
            serialized_inverse,
            *serialized_pooling_inputs,
        )
        bev_features = self.bev_neck(point)
        outputs = self.bbox_head(bev_features)
        return outputs.export_tensors(self.output_names)


class _PTv3Det3DHeadExportModule(nn.Module):
    """Export-only detection head consuming the two coarsest encoder stages."""

    def __init__(
        self,
        bev_neck: PTv3Det3DBEVNeck,
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
        point = rebuild_point_chain([skip_feat, deepest_feat], [cluster], [skip_grid_coord])
        bev = self.bev_neck(point)
        outputs = self.bbox_head(bev)
        return outputs.export_tensors(self.output_names)


class PTv3Detection3DModel(PTv3BaseModel):
    """Compose the PTv3 encoder with dense BEV detection heads."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        bev_neck: PTv3Det3DBEVNeck,
        bbox_head: nn.Module,
        export_output_names: Sequence[str],
        grid_size: float,
        point_cloud_range: Sequence[float],
        log_dict_configs: LogDictConfigs,
        freeze_encoder: bool = False,
        partial_annotations: PartialAnnotations | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 detection model.

        Args:
            encoder: PTv3 encoder module.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            bev_neck: Detection BEV neck consuming the encoder pooling chain.
            bbox_head: Detection head producing the decoded predictions.
            export_output_names: Ordered output names used during export.
            grid_size: Voxel grid size used to derive sparse shape.
            point_cloud_range: Point-cloud range used to derive sparse shape.
            log_dict_configs: Logging configuration for training and validation.
            freeze_encoder: When ``True``, keep the encoder frozen in eval mode.
            partial_annotations: Maps the dataset's per-sample annotation flag to the classes
                the head must not supervise in that sample. None supervises every class.
            **kwargs: Keyword arguments forwarded to :class:`ModuleBaseModel`.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            log_dict_configs=log_dict_configs,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.partial_annotations = partial_annotations
        self.export_output_names = list(export_output_names)

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Raw head outputs returned by :meth:`forward`.

        Returns:
            Flat eval output dict consumed by the detection metric.
        """
        return detection3d_eval_output(self.decode_outputs(batch, outputs), batch)

    def get_export_output_names(self) -> list[str]:
        """Return the ordered export output names."""
        return list(self.export_output_names)

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run PTv3 feature extraction followed by the configured detection head.

        Args:
            multi_task_batch_inputs: Model inputs holding the point cloud and its voxels.

        Returns:
            The outputs of the task heads.
        """
        point = self.encode_batch(multi_task_batch_inputs)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=self.bbox_head(self.bev_neck(point)),
            )
        )

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute detection losses for one batched step.

        Args:
            batch_inputs: Model inputs holding the ground truth boxes.
            outputs: Raw head outputs returned by :meth:`forward`.

        Returns:
            Dictionary of loss terms produced by the detection head.

        Raises:
            ValueError: If the batch carries no detection ground truth.
        """
        gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_detections is None:
            raise ValueError("PTv3 detection losses need the 3D detection ground truth.")
        return self.bbox_head.loss(
            outputs=outputs.detection3d(),
            gt_bboxes_3d=gt_detections.gt_bboxes_3d,
            gt_labels_3d=gt_detections.gt_labels_3d,
            gt_valid_bboxes=gt_detections.gt_valid_bboxes,
            gt_annotated_class_masks=(
                self.partial_annotations.build_annotated_class_masks(
                    gt_detections.gt_traffic_cone_barrier_bbox_status
                )
                if self.partial_annotations is not None
                else None
            ),
        )

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode predictions for inference.

        Args:
            multi_task_batch_inputs: Model inputs of the prediction step, unused.
            outputs: Raw head outputs returned by :meth:`forward`.

        Returns:
            Decoded detector predictions for the current batch.
        """
        del multi_task_batch_inputs
        return self.bbox_head.decode_outputs(outputs=outputs.detection3d())

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the PTv3 detection ONNX export specification.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Deployment export specification for the PTv3 detector.
        """
        inputs = prepare_ptv3_export_inputs(self, batch_inputs)
        input_args, input_names = inputs.encoder_args(SERIALIZED_POOLING_FIELDS)
        export_module = _PTv3Detection3DExportModule(
            encoder=self._prepare_encoder_export(),
            voxel_encoder=self.voxel_encoder,
            bev_neck=deepcopy(self.bev_neck).eval(),
            bbox_head=self.bbox_head.prepare_for_export(
                bev_shape=self.bev_neck.bev_projector.output_shape
            ),
            sparse_shape=inputs.sparse_shape,
            serialized_depth=inputs.serialization_depth,
            output_names=self.export_output_names,
        )
        export_module.eval()
        return ExportSpec(
            module=export_module,
            args=input_args,
            input_names=input_names,
            output_names=self.get_export_output_names(),
            dynamic_axes=build_ptv3_input_dynamic_axes(input_names),
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build split PTv3 detection ONNX export specs for encoder and detection head.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Export specs of the encoder and the detection head.
        """
        context = build_ptv3_export_context(self, batch_inputs)
        return {
            "ptv3_lidar_encoder": build_encoder_export_spec(context),
            "ptv3_det3d_head": build_det3d_head_export_spec(
                context,
                self.bev_neck,
                self.bbox_head.prepare_for_export(
                    bev_shape=self.bev_neck.bev_projector.output_shape
                ),
                self.export_output_names,
            ),
        }
