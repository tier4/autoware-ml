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
