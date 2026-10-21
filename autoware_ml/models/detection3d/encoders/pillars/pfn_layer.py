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

"""Pillar-based layer for pillar-based encoders for LiDAR detection.

This module contains the PFN layer used by PointPillars-style encoders.
"""

from jaxtyping import Float32
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    """Implement one PointPillars PFN layer.

    The layer applies a linear projection, normalization, and pooling over the
    points that belong to each pillar.
    """

    def __init__(self, in_channels: int, out_channels: int, last_layer: bool) -> None:
        """Initialize one PFN layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            last_layer: Whether this is the final PFN stage.
        """
        super().__init__()
        units = out_channels if last_layer else out_channels // 2
        self.last_layer = last_layer
        self.linear = nn.Linear(in_channels, units, bias=False)
        self.norm = nn.BatchNorm1d(units, eps=1e-3, momentum=0.01)

    def forward(
        self, inputs: Float32[torch.Tensor, "num_pillars num_points num_channels"]
    ) -> Float32[torch.Tensor, "num_pillars num_outputs num_output_channels"]:
        """Encode one PFN stage.

        Args:
            inputs: Decorated pillar feature tensor.

        Returns:
            Encoded pillar features. Output shape changes based on whether this is the last PFN stage.
            If it is, the output shape is ``(num_pillars, 1, num_output_channels)``. Otherwise,
            the output shape is ``(num_pillars, num_points, num_output_channels)``.
        """
        x = self.linear(inputs)
        x = self.norm(x.reshape(-1, x.shape[-1])).reshape_as(x)
        x = F.relu(x, inplace=True)
        x_max = x.max(dim=1, keepdim=True).values
        if self.last_layer:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)
