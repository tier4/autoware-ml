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

"""A dense BEV feature map placed in the metric frame, the level a head samples at (x, y)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float


@dataclass(frozen=True)
class BEVLevel:
    """Dense BEV features of one scale with the metric placement of their cells.

    Attributes:
        feat: BEV features ``(B, C, H, W)``, row ``y`` and column ``x`` of the cell grid.
        grid_origin: Metric ``(x, y)`` of the grid corner.
        cell_size: Metric size of a BEV cell.
    """

    feat: Float[torch.Tensor, "batch_size channels height width"]
    grid_origin: tuple[float, float]
    cell_size: float

    def sample(
        self, key_points: Float[torch.Tensor, "batch_size num_queries num_points 3"]
    ) -> Float[torch.Tensor, "batch_size num_queries num_points channels"]:
        """Sample the map bilinearly at the keypoints' (x, y), outside the map reads zeros.

        Args:
            key_points: Metric keypoints ``(B, N, P, 3)``, the height is not read.

        Returns:
            Sampled features ``(B, N, P, C)``.
        """
        height, width = self.feat.shape[-2:]
        extent = key_points.new_tensor([width * self.cell_size, height * self.cell_size])
        origin = key_points.new_tensor(self.grid_origin)
        grid = (key_points[..., :2] - origin) / extent * 2.0 - 1.0
        sampled = F.grid_sample(
            self.feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return sampled.permute(0, 2, 3, 1)
