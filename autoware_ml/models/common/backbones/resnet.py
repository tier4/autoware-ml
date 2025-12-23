# Copyright 2025 TIER IV, Inc.
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

"""ResNet backbone implementations."""

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
    """ResNet18 backbone that outputs spatial feature maps.

    This is a modified ResNet18 that removes the final average pooling
    and fully connected layers, outputting feature maps suitable for
    downstream tasks like classification heads or detection necks.

    Args:
        in_channels: Number of channels in the input tensor.

    Example:
        ```python
        backbone = ResNet18(in_channels=3)
        features = backbone(images)  # [B, 512, H/32, W/32]
        ```
    """

    def __init__(self, in_channels: int) -> None:
        """Initialize ResNet18 backbone.

        Args:
            in_channels: Number of channels in the input tensor.
        """
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        del self.avgpool
        del self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps from input.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature maps of shape (B, 512, H/32, W/32).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
