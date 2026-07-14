# Copyright 2023 OpenMMLab.
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

"""Image necks used by camera-based 3D perception models.

This module provides reusable image necks used to prepare multiview features
for BEV view transforms and fusion models.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedLSSFPN(nn.Module):
    """Top-down neck used by LSS-style image-to-BEV models.

    Higher-resolution image features are combined with upsampled deeper features
    and projected to a unified output width.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        """Initialize the generalized LSS FPN neck.

        Args:
            in_channels: Input channel dimensions for the image feature pyramid.
            out_channels: Output channel dimension for fused features.
            upsample_mode: Interpolation mode used for the top-down pathway.
            align_corners: Whether bilinear interpolation aligns corner pixels.
        """
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("GeneralizedLSSFPN requires at least two feature levels.")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        lateral_convs = []
        output_convs = []
        for level in range(len(self.in_channels) - 1):
            input_width = self.in_channels[level]
            skip_width = (
                self.in_channels[level + 1] if level == len(self.in_channels) - 2 else out_channels
            )
            lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(input_width + skip_width, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Fuse image pyramid features into LSS-friendly feature levels.

        Args:
            inputs: Image feature pyramid ordered from high to low resolution.

        Returns:
            Tuple of fused feature maps passed to LSS-style view transforms.
        """
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} input feature maps, got {len(inputs)}."
            )

        laterals = list(inputs)
        for level in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[level + 1],
                size=laterals[level].shape[-2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners if self.upsample_mode != "nearest" else None,
            )
            fused = torch.cat([laterals[level], upsampled], dim=1)
            laterals[level] = self.output_convs[level](self.lateral_convs[level](fused))
        return tuple(laterals[:-1])
