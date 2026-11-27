# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
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

"""ResNet-vd backbone of RT-DETR, ported from RT-DETRv2 (``src/nn/backbone/presnet.py``).

ResNet-vd (He et al., "Bag of Tricks for Image Classification with Convolutional Neural
Networks") differs from the torchvision ResNet of :mod:`.resnet` in two places:

* the stem is three 3x3 convolutions (stride 2, 1, 1) instead of one 7x7 convolution, and
* a projection shortcut that halves the resolution average-pools first and then applies a
  stride-1 1x1 convolution, instead of a strided 1x1 convolution that skips three of every
  four positions.

Only this "d" variant is kept, since every RT-DETRv2 model uses it. The module names match
the source, so RT-DETR checkpoints load unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from jaxtyping import Float
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.misc import FrozenBatchNorm2d

from autoware_ml.models.common.layers.conv import build_activation, build_conv_norm

_STAGE_BLOCKS: dict[int, tuple[int, int, int, int]] = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}
_STAGE_WIDTHS = (64, 128, 256, 512)
_STAGE_STRIDES = (4, 8, 16, 32)


def _projection_shortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """Build the shortcut of a block that changes the width or the resolution.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride of the block, 1 or 2.

    Returns:
        At stride 1, a 1x1 convolution and batch norm. At stride 2, a 2x2 average pool that
        halves the resolution, followed by the same 1x1 convolution and batch norm at stride 1.
    """
    if stride == 2:
        return nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ("conv", build_conv_norm(in_channels, out_channels, 1, 1)),
                ]
            )
        )
    return build_conv_norm(in_channels, out_channels, 1, stride)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection, the block of ResNet-18 and -34."""

    expansion = 1

    def __init__(
        self, in_channels: int, width: int, stride: int, shortcut: bool, act: str = "relu"
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            width: Output channel count.
            stride: Stride of the first convolution.
            shortcut: Add the input unchanged; otherwise add a projection of it.
            act: Activation name.
        """
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            self.short = _projection_shortcut(in_channels, width, stride)
        self.branch2a = build_conv_norm(in_channels, width, 3, stride, act=act)
        self.branch2b = build_conv_norm(width, width, 3, 1)
        self.act = build_activation(act)

    def forward(
        self, x: Float[Tensor, "batch_size in_channels height width"]
    ) -> Float[Tensor, "batch_size out_channels out_height out_width"]:
        """Apply the block.

        Args:
            x: Input feature map.

        Returns:
            Output feature map, at ``1 / stride`` of the input resolution.
        """
        out = self.branch2b(self.branch2a(x))
        return self.act(out + (x if self.shortcut else self.short(x)))


class BottleNeck(nn.Module):
    """1x1, 3x3 and 1x1 convolutions with a residual connection, the block of ResNet-50+."""

    expansion = 4

    def __init__(
        self, in_channels: int, width: int, stride: int, shortcut: bool, act: str = "relu"
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            width: Width of the 3x3 convolution; the block outputs ``width * 4`` channels.
            stride: Stride of the 3x3 convolution.
            shortcut: Add the input unchanged; otherwise add a projection of it.
            act: Activation name.
        """
        super().__init__()
        out_channels = width * self.expansion
        self.branch2a = build_conv_norm(in_channels, width, 1, 1, act=act)
        self.branch2b = build_conv_norm(width, width, 3, stride, act=act)
        self.branch2c = build_conv_norm(width, out_channels, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            self.short = _projection_shortcut(in_channels, out_channels, stride)
        self.act = build_activation(act)

    def forward(
        self, x: Float[Tensor, "batch_size in_channels height width"]
    ) -> Float[Tensor, "batch_size out_channels out_height out_width"]:
        """Apply the block.

        Args:
            x: Input feature map.

        Returns:
            Output feature map, at ``1 / stride`` of the input resolution.
        """
        out = self.branch2c(self.branch2b(self.branch2a(x)))
        return self.act(out + (x if self.shortcut else self.short(x)))


class Blocks(nn.Module):
    """One stage: a block with a projection shortcut, then blocks with identity shortcuts."""

    def __init__(
        self,
        block: type[BasicBlock] | type[BottleNeck],
        in_channels: int,
        width: int,
        count: int,
        downsample: bool,
        act: str = "relu",
    ) -> None:
        """Initialize the stage.

        Args:
            block: Residual block type.
            in_channels: Input channel count.
            width: Width of the blocks, see the ``width`` of the block type.
            count: Number of blocks.
            downsample: Whether the first block halves the resolution.
            act: Activation name.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    in_channels,
                    width,
                    stride=2 if i == 0 and downsample else 1,
                    shortcut=i != 0,
                    act=act,
                )
            )
            in_channels = width * block.expansion

    def forward(
        self, x: Float[Tensor, "batch_size in_channels height width"]
    ) -> Float[Tensor, "batch_size out_channels out_height out_width"]:
        """Apply every block in turn.

        Args:
            x: Input feature map.

        Returns:
            Output feature map.
        """
        for block in self.blocks:
            x = block(x)
        return x


class PResNet(nn.Module):
    """ResNet-vd backbone that returns the feature maps of selected stages.

    Stage ``i`` (``res_layers[i]``) is at stride ``4 * 2**i`` and has ``64 * 2**i`` channels
    with basic blocks (depth 18 and 34) or four times as many with bottlenecks (depth 50 and
    101). RT-DETR reads the stages ``(1, 2, 3)``, at strides 8, 16 and 32.

    Example:
        ```python
        backbone = PResNet(depth=18)
        c3, c4, c5 = backbone(images)  # 128, 256 and 512 channels at strides 8, 16 and 32
        ```
    """

    def __init__(
        self,
        depth: int,
        out_indices: Sequence[int] = (1, 2, 3),
        act: str = "relu",
        freeze_at: int = -1,
        freeze_norm: bool = False,
    ) -> None:
        """Initialize the backbone.

        Args:
            depth: ResNet depth, one of 18, 34, 50 and 101.
            out_indices: Strictly increasing stage indices in ``0`` to ``3`` whose outputs
                are returned.
            act: Activation name.
            freeze_at: ``-1`` trains every parameter; ``N >= 0`` freezes the stem and the
                first ``N`` stages.
            freeze_norm: Replace every batch norm with a ``FrozenBatchNorm2d``, which keeps
                the statistics and the affine parameters fixed.

        Raises:
            ValueError: If the depth is not supported, or ``out_indices`` is empty, out of
                range or not strictly increasing.
        """
        super().__init__()
        if depth not in _STAGE_BLOCKS:
            raise ValueError(f"Unsupported depth {depth}, expected one of {sorted(_STAGE_BLOCKS)}.")
        indices = tuple(int(i) for i in out_indices)
        if len(indices) == 0 or any(i < 0 or i > 3 for i in indices):
            raise ValueError(f"out_indices must select stages in [0, 3], got {indices}.")
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(f"out_indices must be strictly increasing, got {indices}.")

        block = BottleNeck if depth >= 50 else BasicBlock
        stem_channels = _STAGE_WIDTHS[0]
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1_1", build_conv_norm(3, stem_channels // 2, 3, 2, act=act)),
                    (
                        "conv1_2",
                        build_conv_norm(stem_channels // 2, stem_channels // 2, 3, 1, act=act),
                    ),
                    ("conv1_3", build_conv_norm(stem_channels // 2, stem_channels, 3, 1, act=act)),
                ]
            )
        )

        stage_channels = [width * block.expansion for width in _STAGE_WIDTHS]
        self.res_layers = nn.ModuleList()
        in_channels = stem_channels
        for i, (width, count) in enumerate(zip(_STAGE_WIDTHS, _STAGE_BLOCKS[depth])):
            self.res_layers.append(
                Blocks(block, in_channels, width, count, downsample=i != 0, act=act)
            )
            in_channels = stage_channels[i]

        self.out_indices = indices
        self.out_channels = [stage_channels[i] for i in indices]
        self.out_strides = [_STAGE_STRIDES[i] for i in indices]

        if freeze_at >= 0:
            _freeze_parameters(self.conv1)
            for stage in self.res_layers[:freeze_at]:
                _freeze_parameters(stage)
        if freeze_norm:
            _freeze_batch_norm(self)

    def forward(
        self, x: Float[Tensor, "batch_size 3 height width"]
    ) -> list[Float[Tensor, "batch_size channels out_height out_width"]]:
        """Extract the feature maps of the stages selected by ``out_indices``.

        Args:
            x: Input images.

        Returns:
            One feature map per entry of ``out_indices``, in increasing stage order, with
            ``out_channels`` channels at ``out_strides``.
        """
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        outs = []
        for i, stage in enumerate(self.res_layers):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


def _freeze_parameters(module: nn.Module) -> None:
    """Stop the gradient of every parameter of a module.

    Args:
        module: The module to freeze.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def _freeze_batch_norm(module: nn.Module) -> nn.Module:
    """Replace every ``BatchNorm2d`` of a module tree with a ``FrozenBatchNorm2d``.

    The frozen batch norms start from the defaults, so freeze before loading a checkpoint.

    Args:
        module: Root of the module tree.

    Returns:
        The frozen replacement of ``module`` when it is a batch norm itself, else ``module``
        with its batch norms replaced in place.
    """
    if isinstance(module, nn.BatchNorm2d):
        return FrozenBatchNorm2d(module.num_features, eps=module.eps)
    for name, child in module.named_children():
        frozen = _freeze_batch_norm(child)
        if frozen is not child:
            setattr(module, name, frozen)
    return module
