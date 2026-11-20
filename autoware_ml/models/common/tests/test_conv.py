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

"""Unit tests for the convolution, batch norm and activation block."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from autoware_ml.models.common.layers.conv import ConvModule, build_activation


class TestBuildActivation(unittest.TestCase):
    """The activation lookup."""

    def test_builds_the_named_activation_in_any_case(self) -> None:
        """
        Input: the supported names in lower and RT-DETR's capitalized spelling, and None.
        Expected: the matching modules, and the identity for None.
        Check: the module types.
        """
        for name, expected in (
            ("relu", nn.ReLU),
            ("ReLU", nn.ReLU),
            ("silu", nn.SiLU),
            ("SiLU", nn.SiLU),
            ("gelu", nn.GELU),
            (None, nn.Identity),
        ):
            with self.subTest(name=name):
                self.assertIsInstance(build_activation(name), expected)

    def test_rejects_an_unknown_name(self) -> None:
        """
        Input: an activation name that is not supported.
        Expected: refused instead of falling back to another activation.
        Check: the ValueError.
        """
        with self.assertRaises(ValueError):
            build_activation("mish")


class TestConvModule(unittest.TestCase):
    """The convolution, batch norm and activation block."""

    def test_defaults_are_a_3x3_convolution_and_an_in_place_relu(self) -> None:
        """
        Input: a block built with the defaults, as the 3D detection models build it.
        Expected: a bias-free same-padded 3x3 convolution, the batch norm settings of SECOND
            and an in-place ReLU, as before kernel_size and act were configurable.
        Check: the convolution, batch norm and activation attributes.
        """
        block = ConvModule(4, 6)

        self.assertEqual(block.conv.kernel_size, (3, 3))
        self.assertEqual(block.conv.padding, (1, 1))
        self.assertIsNone(block.conv.bias)
        self.assertEqual((block.norm.eps, block.norm.momentum), (1e-3, 0.01))
        self.assertIsInstance(block.activation, nn.ReLU)
        self.assertTrue(block.activation.inplace)

    def test_same_padding_keeps_the_resolution_at_stride_1_and_halves_it_at_stride_2(
        self,
    ) -> None:
        """
        Input: 1x1, 3x3 and 7x7 kernels at stride 1 and 2 on a 16x16 map.
        Expected: 16x16 at stride 1 and 8x8 at stride 2 for every kernel size.
        Check: the output shapes.
        """
        x = torch.randn(1, 4, 16, 16)
        for kernel_size in (1, 3, 7):
            for stride, size in ((1, 16), (2, 8)):
                with self.subTest(kernel_size=kernel_size, stride=stride):
                    block = ConvModule(4, 6, stride, kernel_size=kernel_size).eval()

                    with torch.no_grad():
                        out = block(x)

                    self.assertEqual(tuple(out.shape), (1, 6, size, size))

    def test_act_none_keeps_negative_outputs(self) -> None:
        """
        Input: a block without activation whose batch norm maps the convolution to its
            negation.
        Expected: the output keeps its negative values.
        Check: the output equals the negated convolution.
        """
        block = ConvModule(4, 4, kernel_size=1, act=None).eval()
        with torch.no_grad():
            block.norm.weight.fill_(-1.0)
            x = torch.rand(1, 4, 5, 5)

            out = block(x)

            expected = -block.conv(x) / (1 + block.norm.eps) ** 0.5
        self.assertIsInstance(block.activation, nn.Identity)
        self.assertTrue((out < 0).any())
        torch.testing.assert_close(out, expected)

    def test_children_follow_rt_detr(self) -> None:
        """
        Input: a block with an activation.
        Expected: the convolution and batch norm parameters under the names of RT-DETR's
            ConvNormLayer, so its checkpoints load unchanged; the activation has none.
        Check: the state dict keys.
        """
        block = ConvModule(4, 6, kernel_size=1, act="silu")

        self.assertEqual(
            sorted(block.state_dict()),
            [
                "conv.weight",
                "norm.bias",
                "norm.num_batches_tracked",
                "norm.running_mean",
                "norm.running_var",
                "norm.weight",
            ],
        )
