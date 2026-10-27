# Copyright 2021 megvii-model. All Rights Reserved.
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

"""Checkpoint-friendly FPN used by the StreamPETR reference recipe.

This is a native port of the StreamPETR ``CPFPN`` neck: plain 1x1 lateral
convolutions (no norm, no activation), a nearest-neighbor top-down pathway,
and a single 3x3 output convolution on the highest-resolution level only.
Unlike :class:`GeneralizedLSSFPN` it is weight-compatible with reference
StreamPETR checkpoints (parameter names mirror the mm ``ConvModule`` layout).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvUnit(nn.Module):
    """Hold one convolution under a ``conv`` attribute (mm ConvModule layout)."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped convolution."""
        return self.conv(x)


class CPFPN(nn.Module):
    """StreamPETR feature pyramid with per-level laterals and top-down adds.

    Outputs one feature map per input level; only the first (highest
    resolution) level passes through a 3x3 refinement convolution, matching
    the reference implementation.
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        """Initialize the CPFPN neck.

        Args:
            in_channels: Input channel dimensions ordered from high to low
                resolution.
            out_channels: Unified output channel dimension.
        """
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList(
            [
                _ConvUnit(nn.Conv2d(channels, out_channels, kernel_size=1))
                for channels in self.in_channels
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [_ConvUnit(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))]
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Xavier-initialize every convolution (reference init)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Fuse the feature pyramid top-down and refine the finest level.

        Args:
            inputs: Feature maps ordered from high to low resolution.

        Returns:
            One fused feature map per input level.
        """
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} input feature maps, got {len(inputs)}."
            )
        laterals = [
            lateral_conv(inputs[level]) for level, lateral_conv in enumerate(self.lateral_convs)
        ]
        for level in range(len(laterals) - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(
                laterals[level], size=laterals[level - 1].shape[2:], mode="nearest"
            )
        outputs = [
            self.fpn_convs[0](laterals[0]) if level == 0 else laterals[level]
            for level in range(len(laterals))
        ]
        return tuple(outputs)
