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

"""Shared layer building blocks for Autoware-ML models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """Convolution, batch normalization, and ReLU activation composed into one block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        transpose: bool = False,
    ) -> None:
        """Initialize the convolution block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Convolution stride.
            transpose: Whether to use transposed convolution.
        """
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=stride,
                stride=stride,
                bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Args:
            x: Input feature map.

        Returns:
            Activated output feature map.
        """
        return self.activation(self.norm(self.conv(x)))
