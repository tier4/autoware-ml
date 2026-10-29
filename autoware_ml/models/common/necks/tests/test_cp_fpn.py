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

"""Unit tests for the CPFPN neck."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from autoware_ml.models.common.necks.cp_fpn import CPFPN


def test_cpfpn_outputs_match_input_levels_and_channels() -> None:
    """One map of ``out_channels`` per input level, at the resolution of that level."""
    neck = CPFPN(in_channels=[24, 40], out_channels=16)
    high = torch.randn(2, 24, 12, 20)
    low = torch.randn(2, 40, 6, 10)
    outputs = neck((high, low))
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 16, 12, 20)
    assert outputs[1].shape == (2, 16, 6, 10)


def test_cpfpn_parameter_names_follow_the_reference_checkpoints() -> None:
    """The parameter names are the reference neck's, so its checkpoints load without a key map.

    One 1x1 lateral convolution per level and a single 3x3 output convolution, each under a
    ``conv`` attribute as in mmcv's ConvModule.
    """
    neck = CPFPN(in_channels=[24, 40], out_channels=16)
    state = neck.state_dict()

    assert sorted(state) == [
        "fpn_convs.0.conv.bias",
        "fpn_convs.0.conv.weight",
        "lateral_convs.0.conv.bias",
        "lateral_convs.0.conv.weight",
        "lateral_convs.1.conv.bias",
        "lateral_convs.1.conv.weight",
    ]
    assert state["lateral_convs.1.conv.weight"].shape == (16, 40, 1, 1)
    assert state["fpn_convs.0.conv.weight"].shape == (16, 16, 3, 3)


def test_cpfpn_adds_the_coarse_level_and_refines_the_finest_level_only() -> None:
    """The finest output adds the nearest-upsampled coarse lateral; the coarse one is left as is.

    Only the finest level passes the 3x3 convolution.
    """
    neck = CPFPN(in_channels=[1, 1], out_channels=1)
    # Unit laterals, the coarse one adding 0.5, and a 3x3 convolution passing its center tap.
    with torch.no_grad():
        for lateral in neck.lateral_convs:
            lateral.conv.weight.fill_(1.0)
            lateral.conv.bias.zero_()
        neck.fpn_convs[0].conv.weight.zero_()
        neck.fpn_convs[0].conv.weight[0, 0, 1, 1] = 1.0
        neck.fpn_convs[0].conv.bias.zero_()
        neck.lateral_convs[1].conv.bias.fill_(0.5)
    high = torch.randn(1, 1, 4, 6)
    low = torch.randn(1, 1, 2, 3)

    fine, coarse = neck((high, low))

    torch.testing.assert_close(coarse, low + 0.5)
    torch.testing.assert_close(fine, high + F.interpolate(low + 0.5, size=(4, 6), mode="nearest"))


def test_cpfpn_rejects_another_number_of_levels() -> None:
    """Every level has its own lateral convolution, so another number of levels is refused."""
    neck = CPFPN(in_channels=[24, 40], out_channels=16)

    with pytest.raises(ValueError, match="Expected 2 input feature maps"):
        neck((torch.randn(1, 24, 12, 20),))
