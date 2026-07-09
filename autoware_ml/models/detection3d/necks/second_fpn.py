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

"""SECOND feature pyramid neck modules.

This module provides the neck components that aggregate multiscale lidar
features for CenterPoint-style detectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.models.common.layers.conv import ConvModule


class SECONDFPN(nn.Module):
    """Fuse multi-scale BEV features in a SECOND-style neck.

    The neck upsamples backbone stages to a common resolution and concatenates
    them along the channel dimension.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        upsample_strides: Sequence[float],
    ) -> None:
        """Initialize the SECOND feature pyramid neck.

        Args:
            in_channels: Input channel dimensions for each backbone stage.
            out_channels: Output channel dimensions for each upsampled stage.
            upsample_strides: Upsampling stride applied to each stage.
        """
        super().__init__()
        blocks = []
        for input_channels, output_channels, stride in zip(
            in_channels, out_channels, upsample_strides
        ):
            if stride >= 1:
                blocks.append(
                    ConvModule(input_channels, output_channels, stride=int(stride), transpose=True)
                )
            else:
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size=int(round(1 / stride)),
                            stride=int(round(1 / stride)),
                            bias=False,
                        ),
                        nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale BEV features.

        Args:
            x: Backbone feature maps ordered from high to low resolution.

        Returns:
            Concatenated BEV feature tensor.
        """
        upsampled = [block(feature) for block, feature in zip(self.blocks, x)]
        return torch.cat(upsampled, dim=1)
