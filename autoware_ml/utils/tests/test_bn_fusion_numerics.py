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

"""Unit tests for the numerics of Conv+BN folding."""

from __future__ import annotations

import logging
import unittest

import torch
from torch import nn

from autoware_ml.utils.bn_fusion import find_conv_bn_pairs, fuse_model_bn

_ATOL = 1e-5


def _randomized_bn(bn: nn.Module, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        bn.weight.copy_(0.5 + torch.rand(bn.num_features, generator=generator))
        bn.bias.copy_(torch.randn(bn.num_features, generator=generator))
        bn.running_mean.copy_(torch.randn(bn.num_features, generator=generator))
        bn.running_var.copy_(0.1 + torch.rand(bn.num_features, generator=generator))
    return bn


def _assert_fusion_matches(case: unittest.TestCase, model: nn.Sequential, x: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        reference = model(x)
        fuse_model_bn(model)
        case.assertIsInstance(model[1], nn.Identity, "BN must be replaced by Identity")
        fused = model(x)
    case.assertTrue(
        torch.allclose(fused, reference, atol=_ATOL),
        f"max abs error {(fused - reference).abs().max().item():.3e} exceeds {_ATOL}",
    )


class TestFusionNumerics(unittest.TestCase):
    """Folding BN into each supported layer type keeps the output within tolerance."""

    def test_conv2d_bn2d(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                torch.manual_seed(0)
                model = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=bias),
                    _randomized_bn(nn.BatchNorm2d(8), seed=1),
                )
                _assert_fusion_matches(self, model, torch.randn(2, 3, 16, 16))

    def test_conv_transpose2d_bn2d(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                # ConvTranspose2d weights are [in, out, H, W]: the BN scale applies to dim 1.
                torch.manual_seed(2)
                model = nn.Sequential(
                    nn.ConvTranspose2d(4, 6, kernel_size=2, stride=2, bias=bias),
                    _randomized_bn(nn.BatchNorm2d(6), seed=3),
                )
                _assert_fusion_matches(self, model, torch.randn(2, 4, 8, 8))

    def test_conv1d_bn1d(self):
        torch.manual_seed(4)
        model = nn.Sequential(
            nn.Conv1d(3, 5, kernel_size=3, padding=1),
            _randomized_bn(nn.BatchNorm1d(5), seed=5),
        )
        _assert_fusion_matches(self, model, torch.randn(2, 3, 12))

    def test_linear_bn1d(self):
        torch.manual_seed(6)
        model = nn.Sequential(
            nn.Linear(7, 9),
            _randomized_bn(nn.BatchNorm1d(9), seed=7),
        )
        _assert_fusion_matches(self, model, torch.randn(4, 7))

    def test_grouped_conv2d_bn2d(self):
        torch.manual_seed(8)
        model = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=4),
            _randomized_bn(nn.BatchNorm2d(8), seed=9),
        )
        _assert_fusion_matches(self, model, torch.randn(2, 8, 10, 10))


class TestContainerReplacement(unittest.TestCase):
    """The folded BN is replaced through the module API, not through item assignment."""

    def test_bn_replacement_works_in_a_container_without_item_assignment(self) -> None:
        """PTv3's ``PointSequential`` names its children by index but has no ``__setitem__``."""

        class NumericContainer(nn.Module):
            applies_children_in_order = True  # what PointSequential declares

            def __init__(self, *children: nn.Module) -> None:
                super().__init__()
                for index, child in enumerate(children):
                    self.add_module(str(index), child)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for child in self._modules.values():
                    x = child(x)
                return x

        torch.manual_seed(0)
        linear = nn.Linear(4, 4)
        norm = nn.BatchNorm1d(4)
        norm.running_mean.normal_()
        norm.running_var.uniform_(0.5, 1.5)
        model = NumericContainer(linear, norm).eval()

        self.assertEqual(find_conv_bn_pairs(model), [("0", "1")])
        x = torch.randn(3, 4)
        expected = model(x)

        fuse_model_bn(model)

        self.assertIsInstance(model.get_submodule("1"), nn.Identity)
        self.assertTrue(torch.allclose(model(x), expected, atol=1e-5))


class _ReluBetween(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.relu(self.conv(x)))


class _DeclaredBlock(nn.Module):
    bn_fusion_pairs = (("conv", "norm"),)

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, 1)
        self.norm = nn.BatchNorm2d(4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


def _randomize_bn(bn: nn.Module) -> None:
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.normal_()


class TestDeclaredPairs(unittest.TestCase):
    """Only the pairs a module declares in ``bn_fusion_pairs`` are folded."""

    def test_adjacent_registration_alone_does_not_authorize_folding(self) -> None:
        """Folding ``bn(relu(conv(x)))`` would move the BN before the ReLU."""

        torch.manual_seed(0)
        model = _ReluBetween().eval()
        _randomize_bn(model.bn)
        x = torch.randn(2, 3, 5, 5)
        expected = model(x)

        with self.assertLogs("autoware_ml.utils.bn_fusion", logging.WARNING) as logs:
            self.assertEqual(find_conv_bn_pairs(model), [])
            fuse_model_bn(model)

        self.assertIsInstance(model.bn, nn.BatchNorm2d, "the undeclared pair must not be folded")
        self.assertTrue(torch.equal(model(x), expected))
        log_text = "\n".join(logs.output)
        self.assertIn("conv + bn", log_text)
        self.assertIn("bn_fusion_pairs", log_text)

    def test_declared_pairs_fold_and_stay_numerically_equivalent(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(_DeclaredBlock(), _DeclaredBlock(in_channels=4)).eval()
        for block in model:
            _randomize_bn(block.norm)
        x = torch.randn(2, 3, 5, 5)
        expected = model(x)

        self.assertEqual(find_conv_bn_pairs(model), [("0.conv", "0.norm"), ("1.conv", "1.norm")])
        fuse_model_bn(model)

        self.assertTrue(all(isinstance(block.norm, nn.Identity) for block in model))
        self.assertTrue(torch.allclose(model(x), expected, atol=1e-5))

    def test_a_wrong_declaration_is_an_error_not_a_skip(self) -> None:
        class Lying(nn.Module):
            bn_fusion_pairs = (("conv", "act"),)

            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.act = nn.ReLU()

        with self.assertRaisesRegex(ValueError, "bn_fusion_pairs names"):
            find_conv_bn_pairs(Lying())
