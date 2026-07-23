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

"""VoVNet backbones reused by camera-based detection models.

This module provides a lightweight native VoVNet implementation.
The exported class follows the same multi-scale tuple interface used
by other Autoware-ML image backbones.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": {
        "stem": [64, 64, 64],
        "stage_conv_ch": [64, 80, 96, 112],
        "stage_out_ch": [112, 256, 384, 512],
        "layer_per_block": 3,
        "block_per_stage": [1, 1, 1, 1],
        "dw": True,
    },
    "V-19-dw-eSE": {
        "stem": [64, 64, 64],
        "stage_conv_ch": [128, 160, 192, 224],
        "stage_out_ch": [256, 512, 768, 1024],
        "layer_per_block": 3,
        "block_per_stage": [1, 1, 1, 1],
        "dw": True,
    },
    "V-19-slim-eSE": {
        "stem": [64, 64, 128],
        "stage_conv_ch": [64, 80, 96, 112],
        "stage_out_ch": [112, 256, 384, 512],
        "layer_per_block": 3,
        "block_per_stage": [1, 1, 1, 1],
        "dw": False,
    },
    "V-19-eSE": {
        "stem": [64, 64, 128],
        "stage_conv_ch": [128, 160, 192, 224],
        "stage_out_ch": [256, 512, 768, 1024],
        "layer_per_block": 3,
        "block_per_stage": [1, 1, 1, 1],
        "dw": False,
    },
    "V-39-eSE": {
        "stem": [64, 64, 128],
        "stage_conv_ch": [128, 160, 192, 224],
        "stage_out_ch": [256, 512, 768, 1024],
        "layer_per_block": 5,
        "block_per_stage": [1, 1, 2, 2],
        "dw": False,
    },
    "V-57-eSE": {
        "stem": [64, 64, 128],
        "stage_conv_ch": [128, 160, 192, 224],
        "stage_out_ch": [256, 512, 768, 1024],
        "layer_per_block": 5,
        "block_per_stage": [1, 1, 4, 3],
        "dw": False,
    },
    "V-99-eSE": {
        "stem": [64, 64, 128],
        "stage_conv_ch": [128, 160, 192, 224],
        "stage_out_ch": [256, 512, 768, 1024],
        "layer_per_block": 5,
        "block_per_stage": [1, 3, 9, 3],
        "dw": False,
    },
}


def _conv3x3(
    in_channels: int,
    out_channels: int,
    module_name: str,
    postfix: str,
    stride: int = 1,
    groups: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
) -> list[tuple[str, nn.Module]]:
    """Build a convolution, batchnorm, ReLU block."""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def _conv1x1(
    in_channels: int, out_channels: int, module_name: str, postfix: str
) -> list[tuple[str, nn.Module]]:
    """Build a 1x1 convolution, batchnorm, ReLU block."""
    return _conv3x3(
        in_channels=in_channels,
        out_channels=out_channels,
        module_name=module_name,
        postfix=postfix,
        kernel_size=1,
        padding=0,
    )


def _dw_conv3x3(
    in_channels: int, out_channels: int, module_name: str, postfix: str, stride: int = 1
) -> list[tuple[str, nn.Module]]:
    """Build a depthwise separable 3x3 block."""
    return [
        (
            f"{module_name}_{postfix}/dw_conv3x3",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
        ),
        (
            f"{module_name}_{postfix}/pw_conv1x1",
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        ),
        (f"{module_name}_{postfix}/pw_norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/pw_relu", nn.ReLU(inplace=True)),
    ]


class _eSEModule(nn.Module):
    """Effective squeeze-excitation block."""

    def __init__(self, channels: int) -> None:
        """Initialize the eSE block."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Re-weight feature responses channel-wise."""
        weight = self.hsigmoid(self.fc(self.avg_pool(x)))
        return x * weight


class _OSAModule(nn.Module):
    """One-Shot Aggregation block used by VoVNet stages."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: int,
        concat_channels: int,
        layers_per_block: int,
        module_name: str,
        identity: bool = False,
        depthwise: bool = False,
    ) -> None:
        """Initialize one OSA module."""
        super().__init__()
        self.identity = identity
        self.depthwise = depthwise
        self.is_reduced = depthwise and in_channels != stage_channels
        self.layers = nn.ModuleList()

        if self.is_reduced:
            self.conv_reduction = nn.Sequential(
                OrderedDict(_conv1x1(in_channels, stage_channels, f"{module_name}_reduction", "0"))
            )

        current_channels = stage_channels if self.is_reduced else in_channels
        for layer_index in range(layers_per_block):
            block = _dw_conv3x3 if depthwise else _conv3x3
            self.layers.append(
                nn.Sequential(
                    OrderedDict(
                        block(current_channels, stage_channels, module_name, str(layer_index))
                    )
                )
            )
            current_channels = stage_channels

        aggregated_channels = in_channels + layers_per_block * stage_channels
        self.concat = nn.Sequential(
            OrderedDict(_conv1x1(aggregated_channels, concat_channels, module_name, "concat"))
        )
        self.ese = _eSEModule(concat_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate intermediate features into one stage output."""
        identity = x
        outputs = [x]
        if self.depthwise and self.is_reduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = self.concat(torch.cat(outputs, dim=1))
        x = self.ese(x)
        if self.identity:
            x = x + identity
        return x


class _OSAStage(nn.Sequential):
    """Stack one or more OSA modules into a VoVNet stage."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: int,
        concat_channels: int,
        blocks_per_stage: int,
        layers_per_block: int,
        stage_num: int,
        depthwise: bool = False,
    ) -> None:
        """Initialize one VoVNet stage."""
        super().__init__()
        if stage_num != 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name,
            _OSAModule(
                in_channels,
                stage_channels,
                concat_channels,
                layers_per_block,
                module_name,
                depthwise=depthwise,
            ),
        )
        for block_index in range(blocks_per_stage - 1):
            module_name = f"OSA{stage_num}_{block_index + 2}"
            self.add_module(
                module_name,
                _OSAModule(
                    concat_channels,
                    stage_channels,
                    concat_channels,
                    layers_per_block,
                    module_name,
                    identity=True,
                    depthwise=depthwise,
                ),
            )


class VoVNetMultiScale(nn.Module):
    """Expose VoVNet intermediate feature maps for downstream tasks."""

    def __init__(
        self,
        spec_name: str,
        input_ch: int = 3,
        out_features: Sequence[str] = ("stage4", "stage5"),
        frozen_stages: int = -1,
        norm_eval: bool = True,
    ) -> None:
        """Initialize the multi-scale VoVNet backbone.

        Args:
            spec_name: VoVNet stage specification name.
            input_ch: Number of channels in the input tensor.
            out_features: Names of feature stages to return.
            frozen_stages: Last stage index to freeze.
            norm_eval: Whether batchnorm layers should stay in eval mode during training.
        """
        super().__init__()
        if spec_name not in _STAGE_SPECS:
            raise ValueError(f"Unsupported VoVNet spec: {spec_name}")
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.out_features = tuple(out_features)

        stage_specs = _STAGE_SPECS[spec_name]
        stem_channels = stage_specs["stem"]
        stage_conv_channels = stage_specs["stage_conv_ch"]
        stage_out_channels = stage_specs["stage_out_ch"]
        blocks_per_stage = stage_specs["block_per_stage"]
        layers_per_block = stage_specs["layer_per_block"]
        depthwise = stage_specs["dw"]

        conv_block = _dw_conv3x3 if depthwise else _conv3x3
        stem = _conv3x3(input_ch, stem_channels[0], "stem", "1", stride=2)
        stem += conv_block(stem_channels[0], stem_channels[1], "stem", "2", stride=1)
        stem += conv_block(stem_channels[1], stem_channels[2], "stem", "3", stride=2)
        self.stem = nn.Sequential(OrderedDict(stem))

        in_channels_per_stage = [stem_channels[2], *stage_out_channels[:-1]]
        self.stage_names: list[str] = []
        for stage_index in range(4):
            name = f"stage{stage_index + 2}"
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSAStage(
                    in_channels=in_channels_per_stage[stage_index],
                    stage_channels=stage_conv_channels[stage_index],
                    concat_channels=stage_out_channels[stage_index],
                    blocks_per_stage=blocks_per_stage[stage_index],
                    layers_per_block=layers_per_block,
                    stage_num=stage_index + 2,
                    depthwise=depthwise,
                ),
            )

    def _freeze_stages(self) -> None:
        """Freeze the configured early stages."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for parameter in self.stem.parameters():
                parameter.requires_grad = False
        for stage_index in range(1, self.frozen_stages + 1):
            stage = getattr(self, f"stage{stage_index + 1}")
            stage.eval()
            for parameter in stage.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> VoVNetMultiScale:
        """Set training mode while honoring stage freezing and norm_eval."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract configured intermediate feature maps.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tuple of selected feature maps ordered by stage.
        """
        outputs = []
        x = self.stem(x)
        if "stem" in self.out_features:
            outputs.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self.out_features:
                outputs.append(x)
        return tuple(outputs)


class VoVNet99MultiScale(VoVNetMultiScale):
    """Expose VoVNet-99 stage outputs for multiview camera models."""

    def __init__(
        self,
        input_ch: int = 3,
        out_features: Sequence[str] = ("stage4", "stage5"),
        frozen_stages: int = -1,
        norm_eval: bool = True,
    ) -> None:
        """Initialize the multi-scale VoVNet-99 backbone."""
        super().__init__(
            spec_name="V-99-eSE",
            input_ch=input_ch,
            out_features=out_features,
            frozen_stages=frozen_stages,
            norm_eval=norm_eval,
        )
