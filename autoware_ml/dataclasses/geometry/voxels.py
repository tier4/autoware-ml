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

"""Container for voxelized point clouds."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Float32, Int32, Int64
import torch


class VoxelsData(NamedTuple):
    """
    Container for hard-voxelization results.

    Attributes:
        voxels (M, max_num_points, C): Padded point features grouped by their respective voxel,
            where a point value is fully 0 when the voxel is padded.
            C is either (x, y, z, intensity) or (x, y, z, time_lag) if C is 4. C is
            (x, y, z, intensity, time_lag) when it's 5.
        coords (M, 3): Integer voxel coordinates in (x, y, z).
        num_points (M): Number of valid points per voxel.
        batch_indices (M): Batch indices for each voxel.
        point_voxel_indices (N): Row in ``voxels`` of every input point, ``-1`` for
            points outside the range or in voxels beyond the ``max_voxels`` budget.
            Points beyond ``max_num_points`` keep their voxel row even though the
            padded features exclude them.
        num_dropped_voxels (): Number of occupied voxels discarded because a sample
            exceeded the ``max_voxels`` budget.
      M = batch_size * maximum number of voxels.
    """

    voxels: Float32[torch.Tensor, "M max_num_points C"]
    coords: Int32[torch.Tensor, "M 3"]
    num_points: Int32[torch.Tensor, " M"]
    batch_indices: Int32[torch.Tensor, " M"]
    point_voxel_indices: Int64[torch.Tensor, " N"]
    num_dropped_voxels: Int64[torch.Tensor, ""]
