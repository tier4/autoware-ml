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

"""BEV levels of a dense BEV map for heads that sample them at metric keypoints."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel


class BEVMapPyramid(nn.Module):
    """BEV levels of the dense map of a lidar branch, placed on the metric grid of the map.

    The map is projected to the level channels and halved into the downsampled levels, so a
    head samples the map at its own resolution and coarser ones.
    """

    def __init__(
        self,
        *,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        out_size_factor: int,
        in_channels: int,
        channels: int,
        num_downsampled: int = 0,
    ) -> None:
        """Initialize the BEV pyramid.

        Args:
            point_cloud_range: Metric range of the voxel grid.
            voxel_size: Metric size of the voxels, square in BEV.
            out_size_factor: BEV downsampling factor between the voxel grid and the map.
            in_channels: Feature dimension of the map.
            channels: Feature dimension of the levels.
            num_downsampled: Levels halved from the projected map.

        Raises:
            ValueError: If the BEV cells are not square.
        """
        super().__init__()
        if voxel_size[0] != voxel_size[1]:
            raise ValueError(f"The BEV cells must be square, got voxel_size {list(voxel_size)}.")
        self.grid_origin = (float(point_cloud_range[0]), float(point_cloud_range[1]))
        cell_size = float(voxel_size[0]) * out_size_factor
        self.cell_sizes = [cell_size * 2**level for level in range(num_downsampled + 1)]
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
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
        self, bev_map: Float32[torch.Tensor, "batch_size channels height width"]
    ) -> list[BEVLevel]:
        """Project the map and build the coarser levels.

        Args:
            bev_map: Dense BEV map of the lidar branch, row ``y`` and column ``x`` of the grid.

        Returns:
            BEV levels from the finest to the coarsest.
        """
        maps = [self.projection(bev_map)]
        for downsample in self.downsample:
            maps.append(downsample(maps[-1]))
        return [
            BEVLevel(feat, self.grid_origin, cell_size)
            for feat, cell_size in zip(maps, self.cell_sizes, strict=True)
        ]
