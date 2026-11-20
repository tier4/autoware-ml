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

"""Unit tests for BatchNorm folding."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from autoware_ml.utils.bn_fusion import find_conv_bn_pairs, fuse_model_bn


def _conv_bn_relu() -> nn.Sequential:
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()).eval()
    # Non-default BatchNorm parameters, so that the fold changes the weights.
    module[1].running_mean.uniform_(-1, 1)
    module[1].running_var.uniform_(0.5, 2.0)
    module[1].weight.data.uniform_(0.5, 1.5)
    module[1].bias.data.uniform_(-0.5, 0.5)
    return module


class _UnorderedBlock(nn.Module):
    """Registers conv then bn, but its forward puts an activation between them."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))


class _DeclaredBlock(_UnorderedBlock):
    bn_fusion_pairs = (("conv", "bn"),)

    def forward(self, x):
        return self.bn(self.conv(x))


class TestFuseModelBn(unittest.TestCase):
    """Folding removes the BatchNorm layers and keeps the module's outputs."""

    def test_fold_removes_bn_and_preserves_outputs(self) -> None:
        module = _conv_bn_relu()
        folded = copy.deepcopy(module)
        fuse_model_bn(folded)

        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in folded.modules()))
        self.assertIsInstance(module[1], nn.BatchNorm2d)
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(folded(x), module(x), rtol=1e-4, atol=1e-5)

    def test_linear_bn1d_and_transposed_conv_fold(self) -> None:
        torch.manual_seed(1)
        module = nn.Sequential(
            nn.Linear(6, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Unflatten(1, (5, 1, 1)),
            nn.ConvTranspose2d(5, 3, 2, stride=2),
            nn.BatchNorm2d(3),
        ).eval()
        for bn in (module[1], module[5]):
            bn.running_mean.uniform_(-1, 1)
            bn.running_var.uniform_(0.5, 2.0)
        self.assertEqual(find_conv_bn_pairs(module), [("0", "1"), ("4", "5")])
        x = torch.randn(4, 6)
        with torch.no_grad():
            expected = module(x)
            fuse_model_bn(module)
            torch.testing.assert_close(module(x), expected, rtol=1e-4, atol=1e-5)


class TestFindConvBnPairs(unittest.TestCase):
    """Pairs are found in order or from a block's declaration, never from registration order."""

    def test_nothing_to_fold_finds_no_pairs(self) -> None:
        module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU()).eval()
        self.assertEqual(find_conv_bn_pairs(module), [])

    def test_registration_order_alone_does_not_pair_but_a_declaration_does(self) -> None:
        with self.assertLogs("autoware_ml.utils.bn_fusion", level="WARNING") as logs:
            self.assertEqual(find_conv_bn_pairs(_UnorderedBlock().eval()), [])
        self.assertIn("bn_fusion_pairs", "\n".join(logs.output))
        self.assertEqual(find_conv_bn_pairs(_DeclaredBlock().eval()), [("conv", "bn")])

    def test_a_wrong_declaration_is_an_error(self) -> None:
        class _Wrong(nn.Module):
            bn_fusion_pairs = (("conv", "missing"),)

            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1)

        with self.assertRaisesRegex(ValueError, "bn_fusion_pairs"):
            find_conv_bn_pairs(_Wrong())
