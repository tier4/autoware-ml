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


def test_cpfpn_has_one_lateral_per_level_and_one_output_convolution() -> None:
    """A 1x1 lateral convolution per level and a single 3x3 convolution for the finest level."""
    neck = CPFPN(in_channels=[24, 40], out_channels=16)

    assert [conv.kernel_size for conv in neck.lateral_convs] == [(1, 1), (1, 1)]
    assert neck.lateral_convs[1].weight.shape == (16, 40, 1, 1)
    assert neck.fpn_conv.weight.shape == (16, 16, 3, 3)
    assert list(neck.state_dict()) == [
        "lateral_convs.0.weight",
        "lateral_convs.0.bias",
        "lateral_convs.1.weight",
        "lateral_convs.1.bias",
        "fpn_conv.weight",
        "fpn_conv.bias",
    ]


def test_cpfpn_adds_the_coarse_level_and_refines_the_finest_level_only() -> None:
    """The finest output is the 3x3 convolution of the fused map; the coarse one is left as is.

    The fused map adds the nearest-upsampled coarse lateral to the fine one. The 3x3 kernel
    is not an identity, so a bypassed or 1x1 output convolution fails the check.
    """
    neck = CPFPN(in_channels=[1, 1], out_channels=1)
    # Unit laterals, the coarse one adding 0.5, and a 3x3 convolution of distinct taps.
    kernel = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3) / 10.0
    with torch.no_grad():
        for lateral in neck.lateral_convs:
            lateral.weight.fill_(1.0)
            lateral.bias.zero_()
        neck.fpn_conv.weight.copy_(kernel)
        neck.fpn_conv.bias.fill_(0.25)
        neck.lateral_convs[1].bias.fill_(0.5)
    high = torch.randn(1, 1, 4, 6)
    low = torch.randn(1, 1, 2, 3)

    fine, coarse = neck((high, low))

    torch.testing.assert_close(coarse, low + 0.5)
    fused = high + F.interpolate(low + 0.5, size=(4, 6), mode="nearest")
    torch.testing.assert_close(fine, F.conv2d(fused, kernel, torch.tensor([0.25]), padding=1))
    assert not torch.allclose(fine, fused)


def test_cpfpn_rejects_another_number_of_levels() -> None:
    """Every level has its own lateral convolution, so another number of levels is refused."""
    neck = CPFPN(in_channels=[24, 40], out_channels=16)

    with pytest.raises(ValueError, match="Expected 2 inputs, got 1"):
        neck((torch.randn(1, 24, 12, 20),))
