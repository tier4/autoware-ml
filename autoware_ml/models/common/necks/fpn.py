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

"""The feature pyramid network of Lin et al., with the semantics of mmdetection's ``FPN``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F

ExtraConvsSource = Literal["on_input", "on_lateral", "on_output"]
"""Where the extra levels of the pyramid are convolved from."""

EXTRA_CONVS_SOURCES: tuple[ExtraConvsSource, ...] = ("on_input", "on_lateral", "on_output")


class FPN(nn.Module):
    """Feature pyramid network.

    Every input level is projected to ``out_channels`` by a 1x1 lateral convolution, the
    deeper levels are upsampled and added top-down, and a 3x3 convolution smooths every
    level. Extra levels beyond the inputs, when ``num_outs`` asks for them, come from
    stride-2 3x3 convolutions on the last output (``on_output``), on the last lateral
    (``on_lateral``) or on the last input (``on_input``), or from max pooling when no extra
    convolutions are asked. The convolutions carry no normalization and no activation, as
    the mmdetection defaults, and start from a Xavier uniform draw.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int | None = None,
        start_level: int = 0,
        add_extra_convs: bool | ExtraConvsSource = False,
        relu_before_extra_convs: bool = False,
        upsample_mode: str = "nearest",
    ) -> None:
        """Initialize the neck.

        Args:
            in_channels: Channels of every input level, from the finest.
            out_channels: Channels of every output level.
            num_outs: Number of output levels, the number of used inputs when ``None``.
            start_level: Index of the first input level used; every deeper level is used.
            add_extra_convs: Where the extra levels are convolved from, ``False`` for max
                pooling. ``True`` selects ``"on_input"`` as mmdetection does.
            relu_before_extra_convs: Whether a ReLU precedes every extra convolution but the
                first.
            upsample_mode: Interpolation of the top-down pathway.

        Raises:
            ValueError: If ``start_level`` is not an input level, ``num_outs`` is too small or
                ``add_extra_convs`` is not a bool or one of the three sources.
        """
        super().__init__()
        if add_extra_convs is True:
            add_extra_convs = "on_input"
        elif add_extra_convs is not False and add_extra_convs not in EXTRA_CONVS_SOURCES:
            raise ValueError(
                f"add_extra_convs={add_extra_convs!r} must be a bool or one of "
                f"{EXTRA_CONVS_SOURCES}."
            )
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        num_ins = len(self.in_channels)
        if not 0 <= start_level < num_ins:
            raise ValueError(f"start_level={start_level} must index one of the {num_ins} inputs.")
        num_used = num_ins - start_level
        if num_outs is None:
            num_outs = num_used
        if num_outs < num_used:
            raise ValueError(
                f"num_outs={num_outs} must cover the {num_used} used input levels; drop input "
                "levels with start_level instead."
            )
        self.start_level = start_level
        self.num_outs = num_outs
        self.add_extra_convs: Literal[False] | ExtraConvsSource = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_mode = upsample_mode

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(self.in_channels[level], out_channels, kernel_size=1)
            for level in range(start_level, num_ins)
        )
        self.fpn_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(num_used)
        )
        num_extra = num_outs - num_used
        if add_extra_convs and num_extra > 0:
            for extra in range(num_extra):
                channels = (
                    self.in_channels[-1]
                    if extra == 0 and add_extra_convs == "on_input"
                    else out_channels
                )
                self.fpn_convs.append(
                    nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
        self.init_weights()

    def init_weights(self) -> None:
        """Draw every convolution from Xavier uniform with a zero bias."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, inputs: Sequence[Float[torch.Tensor, "batch_size channels height width"]]
    ) -> tuple[Float[torch.Tensor, "batch_size out_channels height width"], ...]:
        """Build the pyramid.

        Args:
            inputs: One feature map per input level, from the finest.

        Returns:
            ``num_outs`` feature maps from the finest.

        Raises:
            ValueError: If the number of inputs differs from ``in_channels``.
        """
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} inputs, got {len(inputs)}.")

        laterals = [
            lateral_conv(inputs[self.start_level + level])
            for level, lateral_conv in enumerate(self.lateral_convs)
        ]
        for level in range(len(laterals) - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(
                laterals[level], size=laterals[level - 1].shape[-2:], mode=self.upsample_mode
            )
        outputs = [self.fpn_convs[level](laterals[level]) for level in range(len(laterals))]

        num_extra = self.num_outs - len(outputs)
        if num_extra > 0 and not self.add_extra_convs:
            for _ in range(num_extra):
                outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))
        elif num_extra > 0:
            if self.add_extra_convs == "on_input":
                extra_source = inputs[-1]
            elif self.add_extra_convs == "on_lateral":
                extra_source = laterals[-1]
            else:  # "on_output", the only source left after the __init__ check.
                extra_source = outputs[-1]
            outputs.append(self.fpn_convs[len(laterals)](extra_source))
            for extra in range(1, num_extra):
                source = F.relu(outputs[-1]) if self.relu_before_extra_convs else outputs[-1]
                outputs.append(self.fpn_convs[len(laterals) + extra](source))
        return tuple(outputs)
