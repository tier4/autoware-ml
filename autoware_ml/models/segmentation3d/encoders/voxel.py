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

"""Voxel feature encoders feeding the PTv3 embedding stem."""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanVoxelFeatureEncoder(nn.Module):
    """Reduce padded voxel points to one feature vector per voxel.

    Every point channel is averaged over the valid points of the voxel, so a
    voxel built from several lidar sweeps keeps the mean position, intensity
    and time lag of its points. The module has no parameters; the same graph
    runs in training and in the exported model.
    """

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Average padded voxel points.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)``
                where unused slots are zero.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Voxel features of shape ``(num_voxels, channels)``.
        """
        counts = num_points.to(voxels.dtype).clamp(min=1.0).unsqueeze(1)
        return voxels.sum(dim=1) / counts
