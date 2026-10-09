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

"""Container for a batched point cloud projected into range images."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Int64
import torch


class RangeViewData(NamedTuple):
    """
    Container for the range view projection of a batched point cloud.

    Attributes:
        coors (N, 3): Range image cell of every point as (batch_index, row, column).
        voxel_coors (V, 3): Every occupied range image cell of the batch.
        inverse_map (N): Row in ``voxel_coors`` of the cell every point falls into.
        semantic_labels (B, H, W): Majority label of every range image cell, the ignore index
            for cells without labelled points. None when the batch carries no labels.
      N = number of points, V = number of occupied cells, H and W = range image size.
    """

    coors: Int64[torch.Tensor, "N 3"]
    voxel_coors: Int64[torch.Tensor, "V 3"]
    inverse_map: Int64[torch.Tensor, " N"]
    semantic_labels: Int64[torch.Tensor, "B H W"] | None
