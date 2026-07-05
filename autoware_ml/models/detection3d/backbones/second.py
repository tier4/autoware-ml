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

"""SECOND backbone used by LiDAR detection models.

This module contains the reusable 2D CNN backbone used by PointPillars-style detectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.models.common.layers.conv import ConvModule


class SECONDBackbone(nn.Module):
    """Implement the 2D CNN backbone used by PointPillars-style detectors.

    The backbone processes dense BEV features through staged convolutions and
    exposes multi-scale outputs for downstream necks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        layer_nums: Sequence[int],
        layer_strides: Sequence[int],
    ) -> None:
        """Initialize the SECOND backbone.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel counts per stage.
            layer_nums: Number of residual layers per stage.
            layer_strides: Stride applied by each stage.
        """
        super().__init__()
        blocks = []
        current_channels = in_channels
        for stage_channels, num_layers, stride in zip(out_channels, layer_nums, layer_strides):
            layers = [ConvModule(current_channels, stage_channels, stride=stride)]
            layers.extend(ConvModule(stage_channels, stage_channels) for _ in range(num_layers))
            blocks.append(nn.Sequential(*layers))
            current_channels = stage_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale BEV features.

        Args:
            x: Input BEV feature map.

        Returns:
            List of multi-scale BEV feature maps.
        """
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs
