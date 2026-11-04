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

"""Fused sparse stages of the PTv3 encoder the SparseDrive detection head samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float, Integer

from autoware_ml.dataclasses.models.detection3d.bev_level import BEVLevel
from autoware_ml.models.detection3d.ptv3 import PTv3BEVEncoder, PTv3BEVProjection
from autoware_ml.utils.point_cloud.structures import Point


def _linear_bn_gelu(in_channels: int, out_channels: int) -> nn.Sequential:
    """Return a point-wise projection with batch norm and GELU."""
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.GELU(),
    )


class SparseStage(NamedTuple):
    """Features, voxel coordinates and batch offsets of one fused sparse encoder stage.

    Attributes:
        feat: Feature of every voxel of the batch.
        grid_coord: Voxel coordinates ``(x, y, z)`` of the stage grid.
        offset: Cumulative voxel count of the samples.
    """

    feat: Float[torch.Tensor, "num_voxels channels"]
    grid_coord: Integer[torch.Tensor, "num_voxels 3"]
    offset: Integer[torch.Tensor, " batch_size"]


class PTv3Det3DFeaturePyramid(nn.Module):
    """Fuse the deepest encoder stage top-down into the finer stages and keep every fused stage.

    Every step mirrors one decoder unpooling step, reading the pooling chain without changing it.
    """

    def __init__(self, stage_channels: Sequence[int], out_channels: int) -> None:
        """Initialize the feature pyramid.

        Args:
            stage_channels: Feature dimensions of the stages from the finest fused one to the
                deepest one.
            out_channels: Fused feature dimension.

        Raises:
            ValueError: If fewer than two stages are given.
        """
        super().__init__()
        if len(stage_channels) < 2:
            raise ValueError(f"The pyramid fuses at least two stages, got {list(stage_channels)}.")
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
        """Number of returned levels, every stage but the deepest."""
        return len(self.proj_skip)

    def forward(self, point: Point) -> list[SparseStage]:
        """Fuse the deepest stage down to the finest fused stage.

        Args:
            point: Deepest encoder point with its pooling chain attached.

        Returns:
            The fused stages from the finest to the coarsest.
        """
        levels: list[SparseStage] = []
        child, coarse = point, self.proj_coarse[0](point.feat)
        # The fused stage of every step is projected for the next finer one, the last is not used
        next_projections = (*self.proj_coarse[1:], nn.Identity())
        for proj_skip, proj_next in zip(reversed(self.proj_skip), next_projections, strict=True):
            parent = child["pooling_parent"]
            fused = proj_skip(parent.feat) + coarse[child["pooling_inverse"]]
            levels.append(
                SparseStage(feat=fused, grid_coord=parent.grid_coord, offset=parent.offset)
            )
            child, coarse = parent, proj_next(fused)
        return levels[::-1]


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
        self.channels = channels
        cells = [
            round((point_cloud_range[axis + 3] - point_cloud_range[axis]) / grid_size)
            for axis in (0, 1)
        ]
        self.stage_strides: list[int] = list(stage_strides)
        self.stage_indices: list[int] = [
            self.stage_strides.index(stride) for stride in projected_strides
        ]
        projected_cells = [grid_size * stride for stride in projected_strides]
        downsampled_cells = [
            projected_cells[-1] * 2 ** (level + 1) for level in range(num_downsampled)
        ]
        self.cell_sizes: list[float] = projected_cells + downsampled_cells
        # A stride 2 convolution centers coarse cell j on fine cell 2j, so its grid starts earlier
        shifts = [0.0] * len(projected_cells) + [
            (cell_size - projected_cells[-1]) / 2 for cell_size in downsampled_cells
        ]
        self.grid_origins: list[tuple[float, float]] = [
            (point_cloud_range[0] - shift, point_cloud_range[1] - shift) for shift in shifts
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

    def forward(self, stages: Sequence[SparseStage]) -> list[BEVLevel]:
        """Project the stages onto BEV and build the coarser levels.

        Args:
            stages: Features, voxel coordinates and batch offsets of the fused stages, finest
                first.

        Returns:
            BEV levels from the finest to the coarsest.

        Raises:
            ValueError: If the stages are not the fused stages of the configured strides.
        """
        if len(stages) != len(self.stage_strides):
            raise ValueError(
                f"The pyramid projects stages of the strides {self.stage_strides}, but got "
                f"{len(stages)} stages."
            )
        maps = [
            projection(stages[index].feat, stages[index].grid_coord, stages[index].offset)
            for projection, index in zip(self.projections, self.stage_indices, strict=True)
        ]
        maps[-1] = self.bev_encoder(maps[-1])
        for downsample in self.downsample:
            maps.append(downsample(maps[-1]))
        return [
            BEVLevel(feat, grid_origin, cell_size)
            for feat, grid_origin, cell_size in zip(
                maps, self.grid_origins, self.cell_sizes, strict=True
            )
        ]
