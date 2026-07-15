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

"""Feature fusion modules for detection3d models.

This module provides reusable BEV feature fusion layers shared by 3D detection
architectures.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class ConvFuser(nn.Module):
    """Fuse BEV feature maps by concatenation followed by projection.

    The module merges multiple BEV feature branches into one fused tensor with
    a single convolutional projection.
    """

    def __init__(
        self, in_channels: Sequence[int], out_channels: int, kernel_size: int = 3, padding: int = 1
    ) -> None:
        """Initialize the convolutional BEV fuser.

        Args:
            in_channels: Input channel counts for each branch.
            out_channels: Output channel count after fusion.
            kernel_size: Convolution kernel size.
            padding: Convolution padding.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                sum(in_channels), out_channels, kernel_size=kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse BEV feature maps from multiple branches.

        Args:
            inputs: Sequence of BEV feature tensors with matching spatial shapes.

        Returns:
            Fused BEV feature map.
        """
        return self.proj(torch.cat(list(inputs), dim=1))
