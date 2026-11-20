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


def build_activation(name: str | None) -> nn.Module:
    """Build an activation module by its name.

    Args:
        name: ``relu``, ``silu`` or ``gelu``, in any case, or ``None`` for no activation.
            ReLU and SiLU are built in place.

    Returns:
        The activation, ``nn.Identity`` for ``None``.

    Raises:
        ValueError: If the name is none of the supported activations.
    """
    if name is None:
        return nn.Identity()
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "silu":
        return nn.SiLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation {name!r}, expected relu, silu, gelu or None.")


class ConvModule(nn.Module):
    """Convolution, batch normalization, and activation composed into one block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        transpose: bool = False,
        norm_eps: float = 1e-3,
        norm_momentum: float = 0.01,
        kernel_size: int = 3,
        act: str | None = "relu",
    ) -> None:
        """Initialize the convolution block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Convolution stride.
            transpose: Whether to use transposed convolution, whose kernel size is the stride.
            norm_eps: Epsilon used by the batch normalization layer.
            norm_momentum: Momentum used by the batch normalization layer.
            kernel_size: Kernel size of the regular convolution, padded by
                ``(kernel_size - 1) // 2`` to keep the resolution at stride 1. Ignored when
                ``transpose`` is set.
            act: Activation name, see :func:`build_activation`, or ``None`` for none.
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
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        self.norm = nn.BatchNorm2d(out_channels, eps=norm_eps, momentum=norm_momentum)
        self.activation = build_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Args:
            x: Input feature map.

        Returns:
            Activated output feature map.
        """
        return self.activation(self.norm(self.conv(x)))
