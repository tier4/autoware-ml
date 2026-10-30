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

"""Unit tests for the feature pyramid network neck."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from autoware_ml.models.common.necks.fpn import FPN

IN_CHANNELS = [8, 16, 32, 64]
OUT_CHANNELS = 4


def _pyramid(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    """A four-level pyramid at strides 4, 8, 16 and 32 of a 64x96 image."""
    torch.manual_seed(0)
    return (
        torch.randn(batch_size, 8, 16, 24),
        torch.randn(batch_size, 16, 8, 12),
        torch.randn(batch_size, 32, 4, 6),
        torch.randn(batch_size, 64, 2, 3),
    )


def _set_identity_pyramid(neck: FPN) -> None:
    """Make every lateral conv pass the first out_channels inputs and every 3x3 conv the center tap."""
    with torch.no_grad():
        for lateral in neck.lateral_convs:
            lateral.weight.zero_()
            lateral.bias.zero_()
            for channel in range(OUT_CHANNELS):
                lateral.weight[channel, channel, 0, 0] = 1.0
        for conv in neck.fpn_convs:
            conv.weight.zero_()
            conv.bias.zero_()
            for channel in range(OUT_CHANNELS):
                conv.weight[channel, channel, 1, 1] = 1.0


class TestFPN(unittest.TestCase):
    """The feature pyramid network."""

    def test_returns_one_output_per_input_at_the_input_resolutions(self) -> None:
        """
        Input: the four-level pyramid and a neck of four outputs.
        Expected: four maps of out_channels at the input resolutions.
        Check: the output shapes.
        """
        neck = FPN(IN_CHANNELS, OUT_CHANNELS, num_outs=4).eval()

        outputs = neck(_pyramid())

        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[0].shape, (2, OUT_CHANNELS, 16, 24))
        self.assertEqual(outputs[1].shape, (2, OUT_CHANNELS, 8, 12))
        self.assertEqual(outputs[2].shape, (2, OUT_CHANNELS, 4, 6))
        self.assertEqual(outputs[3].shape, (2, OUT_CHANNELS, 2, 3))

    def test_top_down_pathway_adds_the_upsampled_deeper_level(self) -> None:
        """
        Input: a two-level pyramid whose convs are set to identities, the coarse level
            holding 1 and the fine level 2 on the first channel.
        Expected: the fine output is 2 + 1 = 3 everywhere, the coarse output 1, since the
            coarse lateral is nearest-upsampled and added before the 3x3 conv.
        Check: the two outputs.
        """
        neck = FPN([8, 16], OUT_CHANNELS).eval()
        _set_identity_pyramid(neck)
        fine = torch.zeros(1, 8, 4, 4)
        fine[:, 0] = 2.0
        coarse = torch.zeros(1, 16, 2, 2)
        coarse[:, 0] = 1.0

        outputs = neck((fine, coarse))

        torch.testing.assert_close(outputs[0][0, 0], torch.full((4, 4), 3.0))
        torch.testing.assert_close(outputs[1][0, 0], torch.full((2, 2), 1.0))
        torch.testing.assert_close(outputs[0][0, 1], torch.zeros(4, 4))

    def test_start_level_drops_the_finest_inputs(self) -> None:
        """
        Input: the four-level pyramid and ``start_level=1`` without ``num_outs``.
        Expected: three outputs from the stride-8 level, and only three lateral convs.
        Check: the output count, the first resolution and the lateral count.
        """
        neck = FPN(IN_CHANNELS, OUT_CHANNELS, start_level=1).eval()

        outputs = neck(_pyramid())

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, (2, OUT_CHANNELS, 8, 12))
        self.assertEqual(len(neck.lateral_convs), 3)

    def test_extra_levels_by_max_pooling(self) -> None:
        """
        Input: the pyramid and ``num_outs=5`` without extra convs.
        Expected: a fifth level that is the fourth strided by two, with no extra conv.
        Check: the fifth shape, its values and the conv count.
        """
        neck = FPN(IN_CHANNELS, OUT_CHANNELS, num_outs=5).eval()

        outputs = neck(_pyramid())

        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[4].shape, (2, OUT_CHANNELS, 1, 2))
        torch.testing.assert_close(outputs[4], outputs[3][:, :, ::2, ::2])
        self.assertEqual(len(neck.fpn_convs), 4)

    def test_extra_levels_by_convolution(self) -> None:
        """
        Input: ``num_outs=6`` with extra convs on the output and a ReLU before the second,
            on the pyramid.
        Expected: two stride-2 convs are appended, the first reading out_channels, and the
            sixth level is the fifth convolved after a ReLU.
        Check: the conv count and strides, the sixth shape and its value.
        """
        neck = FPN(
            IN_CHANNELS,
            OUT_CHANNELS,
            num_outs=6,
            add_extra_convs="on_output",
            relu_before_extra_convs=True,
        ).eval()

        outputs = neck(_pyramid())

        self.assertEqual(len(neck.fpn_convs), 6)
        self.assertEqual(neck.fpn_convs[4].stride, (2, 2))
        self.assertEqual(neck.fpn_convs[4].in_channels, OUT_CHANNELS)
        self.assertEqual(outputs[5].shape, (2, OUT_CHANNELS, 1, 1))
        torch.testing.assert_close(outputs[5], neck.fpn_convs[5](torch.relu(outputs[4])))

    def test_extra_conv_on_input_reads_the_last_input(self) -> None:
        """
        Input: ``num_outs=5`` with the extra conv on the input.
        Expected: the extra conv reads the 64 channels of the last input.
        Check: its input width and the fifth level equals that conv on the last input.
        """
        neck = FPN(IN_CHANNELS, OUT_CHANNELS, num_outs=5, add_extra_convs="on_input").eval()
        pyramid = _pyramid()

        outputs = neck(pyramid)

        self.assertEqual(neck.fpn_convs[4].in_channels, 64)
        torch.testing.assert_close(outputs[4], neck.fpn_convs[4](pyramid[3]))

    def test_true_selects_the_extra_conv_on_input(self) -> None:
        """
        Input: ``num_outs=5`` with ``add_extra_convs=True``, as mmdetection configs write it.
        Expected: the same neck as ``"on_input"``: the extra conv reads the 64 channels of
            the last input and the fifth level is that conv on the last input.
        Check: the stored source, the conv's input width and the fifth level.
        """
        neck = FPN(IN_CHANNELS, OUT_CHANNELS, num_outs=5, add_extra_convs=True).eval()
        pyramid = _pyramid()

        outputs = neck(pyramid)

        self.assertEqual(neck.add_extra_convs, "on_input")
        self.assertEqual(neck.fpn_convs[4].in_channels, 64)
        torch.testing.assert_close(outputs[4], neck.fpn_convs[4](pyramid[3]))

    def test_rejects_an_unknown_extra_conv_source(self) -> None:
        """
        Input: a misspelled extra conv source.
        Expected: refused at construction instead of silently convolving the last output.
        Check: ValueError naming the value.
        """
        with self.assertRaisesRegex(ValueError, "on_inputs"):
            FPN(IN_CHANNELS, OUT_CHANNELS, num_outs=5, add_extra_convs="on_inputs")

    def test_weights_start_from_xavier_with_zero_bias(self) -> None:
        """
        Input: a fresh neck.
        Expected: zero biases and weights within the Xavier uniform bound of the lateral
            conv, sqrt(6 / (fan_in + fan_out)).
        Check: the bias and the weight range of the first lateral conv.
        """
        neck = FPN(IN_CHANNELS, OUT_CHANNELS)

        lateral = neck.lateral_convs[0]
        torch.testing.assert_close(lateral.bias, torch.zeros(OUT_CHANNELS))
        bound = math.sqrt(6.0 / 12.0)
        self.assertLessEqual(lateral.weight.abs().max().item(), bound)

    def test_rejects_bad_levels(self) -> None:
        """
        Input: a start level past the end, fewer outputs than inputs, and a wrong input count.
        Expected: all three are refused.
        Check: ValueError for each.
        """
        with self.assertRaisesRegex(ValueError, "start_level"):
            FPN(IN_CHANNELS, OUT_CHANNELS, start_level=4)
        with self.assertRaisesRegex(ValueError, "num_outs"):
            FPN(IN_CHANNELS, OUT_CHANNELS, num_outs=3)
        with self.assertRaisesRegex(ValueError, "Expected 4 inputs"):
            FPN(IN_CHANNELS, OUT_CHANNELS)(_pyramid()[:3])

    def test_parameter_names(self) -> None:
        """
        Input: a four-level neck.
        Expected: the convs are named lateral_convs and fpn_convs by level.
        Check: the state dict keys of the first level.
        """
        keys = set(FPN(IN_CHANNELS, OUT_CHANNELS).state_dict())

        self.assertIn("lateral_convs.0.weight", keys)
        self.assertIn("fpn_convs.3.bias", keys)
        self.assertIsInstance(FPN(IN_CHANNELS, OUT_CHANNELS).fpn_convs[0], nn.Conv2d)
