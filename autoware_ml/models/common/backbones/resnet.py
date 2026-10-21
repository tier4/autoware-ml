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

"""ResNet backbone implementations.

This module contains reusable ResNet backbones for image-based models.
"""

from jaxtyping import Float32
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class _ResNetStages(nn.Module):
    """Stem and residual stages of a torchvision ResNet, without the classification head.

    The stages are built by torchvision and taken over as attributes of this module, so the
    parameter names (``conv1``, ``bn1``, ``layer1`` to ``layer4``) match a plain torchvision
    ResNet and its checkpoints, while the average pooling and the classifier are never created.

    The ``block`` type decides how each residual block is built and how wide every stage is.
    Both blocks add their input back through a skip connection before the final ReLU, and
    both downsample on the first block of a stage. ``width`` is the internal width torchvision
    assigns to the stage, 64, 128, 256 and 512 for ``layer1`` to ``layer4``::

        BasicBlock (ResNet-18/34)              Bottleneck (ResNet-50/101/152)
        expansion = 1                          expansion = 4

        x --------------------+                x ----------------------------+
        |                     |                |                             |
        conv 3x3, width       |                conv 1x1, width  (reduce)     |
        BN + ReLU             |                BN + ReLU                     |
        |                     |                |                             |
        conv 3x3, width       |                conv 3x3, width  (stride)     |
        BN                    |                BN + ReLU                     |
        |                     |                |                             |
        +<--------------------+                conv 1x1, width * 4 (expand)  |
        |                                      BN                            |
        ReLU                                   |                             |
        |                                      +<----------------------------+
        out: width channels                    |
                                               ReLU
                                               |
                                               out: width * 4 channels

    The skip path is the identity, or a 1x1 strided convolution with batch norm when the
    stage changes the resolution or the width. Because of the expansion, the stage outputs
    are ``64, 128, 256, 512`` channels with basic blocks and ``256, 512, 1024, 2048`` with
    bottlenecks, at the same strides 4, 8, 16 and 32. Pretrained weights of one block type
    cannot be loaded into the other.
    """

    def __init__(
        self, block: type[BasicBlock] | type[Bottleneck], layers: list[int], in_channels: int
    ) -> None:
        """Initialize the ResNet stages.

        Args:
            block: Residual block type.
            layers: Number of residual blocks per stage.
            in_channels: Number of channels in the input tensor.
        """
        super().__init__()
        resnet = ResNet(block=block, layers=layers)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward_stem(
        self, x: Float32[Tensor, "batch_size in_channels height width"]
    ) -> Float32[Tensor, "batch_size 64 height/4 width/4"]:
        """Run the stem convolution, normalization, activation and max pooling.

        Args:
            x: Input tensor.

        Returns:
            Stem feature maps at stride 4.
        """
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))


class ResNet18(_ResNetStages):
    """ResNet18 backbone that outputs spatial feature maps.

    This is a ResNet18 without the final average pooling and fully connected layers,
    outputting feature maps suitable for downstream tasks like classification heads or
    detection necks.

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
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], in_channels=in_channels)

    def forward(
        self, x: Float32[Tensor, "batch_size in_channels height width"]
    ) -> Float32[Tensor, "batch_size 512 height/32 width/32"]:
        """Extract spatial feature maps from input.

        Args:
            x: Input tensor.

        Returns:
            Feature maps of the last stage at stride 32.
        """
        x = self.forward_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class _ResNetMultiScale(_ResNetStages):
    """Expose a torchvision ResNet as a reusable multi-scale backbone.

    The wrapper returns intermediate feature maps from every residual stage instead
    of only the final classification features.
    """

    def forward(
        self, x: Float32[Tensor, "batch_size in_channels height width"]
    ) -> tuple[
        Float32[Tensor, "batch_size c2_channels height/4 width/4"],
        Float32[Tensor, "batch_size c3_channels height/8 width/8"],
        Float32[Tensor, "batch_size c4_channels height/16 width/16"],
        Float32[Tensor, "batch_size c5_channels height/32 width/32"],
    ]:
        """Extract feature maps from ``layer1`` through ``layer4``.

        Args:
            x: Input tensor.

        Returns:
            Tuple of multi-scale feature maps ``(c2, c3, c4, c5)`` at strides 4, 8, 16 and 32.
            The channel widths depend on the block type, 64 to 512 for basic blocks and
            256 to 2048 for bottlenecks.
        """
        x = self.forward_stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class ResNet18MultiScale(_ResNetMultiScale):
    """Expose ResNet-18 intermediate feature maps for downstream tasks.

    This wrapper is used by image and multiview models that need feature
    pyramids rather than ImageNet classification logits.
    """

    def __init__(self, in_channels: int) -> None:
        """Initialize the multi-scale ResNet18 backbone.

        Args:
            in_channels: Number of channels in the input tensor.
        """
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], in_channels=in_channels)


class ResNet50MultiScale(_ResNetMultiScale):
    """Expose ResNet-50 intermediate feature maps for downstream tasks.

    This wrapper is used by image and multiview models that need feature
    pyramids rather than ImageNet classification logits.
    """

    def __init__(self, in_channels: int) -> None:
        """Initialize the multi-scale ResNet50 backbone.

        Args:
            in_channels: Number of channels in the input tensor.
        """
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], in_channels=in_channels)
