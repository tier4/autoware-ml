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

"""Container for a point cloud subsampled to one point per voxel."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Int32, Int64
import torch


class GridSampleData(NamedTuple):
    """
    Container for the grid subsampling of a batched point cloud.

    Every occupied voxel of every sample keeps one representative point. The inverse index maps
    every input point back to its voxel, so predictions made on the representatives reach all
    the points.

    Attributes:
        grid_coords (V, 3): Integer voxel coordinate of every representative, counted from the
            minimum of the point cloud range.
        representative_indices (V): Row of every representative in the input points.
        inverse (N): Representative row of the voxel every input point falls into.
        offsets (B): Cumulative representative count of every sample.
      V = number of occupied voxels of the batch, N = number of input points.
    """

    grid_coords: Int32[torch.Tensor, "V 3"]
    representative_indices: Int64[torch.Tensor, " V"]
    inverse: Int64[torch.Tensor, " N"]
    offsets: Int64[torch.Tensor, " B"]
