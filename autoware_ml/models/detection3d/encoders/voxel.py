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

"""Voxel feature encoders for sparse lidar detectors."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class HardSimpleVoxelSinCosEncoder(nn.Module):
    """Mean-pool voxel encoder with Fourier (sin/cos) feature expansion.

    Each voxel's points are mean-pooled, every (min/max-normalized) channel is
    expanded with a set of frequencies, and ``cos``/``sin`` of the result are
    concatenated.

    Args:
        min_norm_values: Per-channel minimum used for normalization (length C).
        max_norm_values: Per-channel maximum used for normalization (length C).
        in_channels: Number of raw point feature channels C.
    """

    def __init__(
        self,
        min_norm_values: Sequence[float],
        max_norm_values: Sequence[float],
        in_channels: int = 4,
    ) -> None:
        super().__init__()
        if len(min_norm_values) != in_channels or len(max_norm_values) != in_channels:
            raise ValueError(
                "min_norm_values and max_norm_values must each have length in_channels="
                f"{in_channels}, got {len(min_norm_values)} and {len(max_norm_values)}."
            )
        self.in_channels = in_channels

        # Fold ((x - min) / (max - min)) * pi * 2^k into y = scale * x + bias so
        # the per-channel Fourier expansion is a single affine op.
        min_values = torch.tensor(min_norm_values, dtype=torch.float32)
        max_values = torch.tensor(max_norm_values, dtype=torch.float32)
        exponents = (2 ** torch.arange(0, in_channels)).float()
        alpha = (math.pi * exponents).unsqueeze(0)  # (1, C) frequencies
        beta = (max_values - min_values).unsqueeze(1)  # (C, 1) range per channel
        scale = alpha / beta  # (C, C)
        bias = -(alpha * min_values.unsqueeze(1)) / beta  # (C, C)
        self.register_buffer("exponent_scale", scale.unsqueeze(0), persistent=False)  # (1, C, C)
        self.register_buffer("exponent_bias", bias.unsqueeze(0), persistent=False)  # (1, C, C)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode padded voxel points into Fourier voxel features.

        Args:
            voxels: Padded voxel points of shape ``(N, max_points, C)``.
            num_points: Number of valid points per voxel of shape ``(N,)``.
            coords: Voxel coordinates with batch index (unused, kept for the
                shared voxel-encoder interface).

        Returns:
            Voxel features of shape ``(N, 2 * C ** 2)``.
        """
        del coords
        voxel_mean = (
            voxels.sum(dim=1, keepdim=False) / num_points.type_as(voxels).clamp(min=1.0).view(-1, 1)
        ).contiguous()
        # (1, C, C) + (1, C, C) * (N, C, 1) -> (N, C, C)
        y = torch.addcmul(self.exponent_bias, self.exponent_scale, voxel_mean.unsqueeze(-1))
        y = y.reshape(-1, self.in_channels * self.in_channels)
        return torch.cat([torch.cos(y), torch.sin(y)], dim=1)
