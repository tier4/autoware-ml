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

"""Tests for the submanifold convolution layer and its choice of library."""

from __future__ import annotations

import unittest
import warnings

import spconv.pytorch as spconv
import torch
from spconv.pytorch.core import SparseConvTensor

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    is_sparse_conv_module,
    replace_submconv3d_for_export,
)
from autoware_ml.ops import submanifold_conv
from autoware_ml.ops.spconv.sparse_conv import ExportableSubMConv3d
from autoware_ml.ops.submanifold_conv import SpconvSubMConv3d, SubMConv3d
from autoware_ml.utils.point_cloud.structures import Point

CUDA = torch.cuda.is_available()
T4SPARSE_ON_CUDA = CUDA and submanifold_conv.IS_T4SPARSE_AVAILABLE


def make_sparse_tensor(
    num_points: int, channels: int, device: torch.device, seed: int = 0
) -> SparseConvTensor:
    """Return a two-sample sparse tensor of random features on a 48^3 grid."""
    generator = torch.Generator().manual_seed(seed)
    coords = torch.unique(torch.randint(0, 48, (num_points, 3), generator=generator), dim=0)
    batch = (torch.arange(coords.shape[0]) >= coords.shape[0] // 2).int().unsqueeze(1)
    indices = torch.cat([batch, coords.int()], dim=1).to(device)
    features = torch.randn(coords.shape[0], channels, generator=generator).to(device)
    return SparseConvTensor(features, indices, [48, 48, 48], 2)


def make_point(num_points: int, channels: int, device: torch.device, seed: int = 0) -> Point:
    """Return a two-sample point container, sparsified, on a 64^3 grid."""
    generator = torch.Generator().manual_seed(seed)
    grid_coord = torch.unique(torch.randint(0, 64, (num_points, 3), generator=generator), dim=0)
    count = grid_coord.shape[0]
    point = Point(
        feat=torch.randn(count, channels, generator=generator).to(device),
        coord=grid_coord.float().to(device),
        grid_coord=grid_coord.to(device),
        offset=torch.tensor([count // 2, count], dtype=torch.long, device=device),
    )
    point.sparsify()
    return point


def conv_only_block(channels: int) -> Block:
    """A PTv3 block reduced to its positional-encoding convolution."""
    return Block(
        channels=channels,
        num_heads=2,
        patch_size=64,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        order_index=0,
        cpe_indice_key="stage0",
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enable_attn=False,
    )


def cpe_conv(block: Block) -> torch.nn.Module:
    """The sparse convolution at the front of a block's positional encoding."""
    return block.cpe._modules["0"]


@unittest.skipUnless(CUDA, "spconv's kernels need a CUDA device")
class TestSpconvLayer(unittest.TestCase):
    """The spconv flavour runs outside autocast, in the dtype of its weights."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.layer = SpconvSubMConv3d(32, 32, kernel_size=3, indice_key="k").to(self.device)
        self.x = make_sparse_tensor(3000, 32, self.device)

    def test_runs_in_fp32_under_autocast(self) -> None:
        with torch.autocast("cuda", dtype=torch.float16):
            y = self.layer(self.x)
        self.assertEqual(y.features.dtype, torch.float32)

    def test_casts_halved_features_to_the_weight_dtype(self) -> None:
        halved = self.x.replace_feature(self.x.features.half())
        self.layer.eval()
        with torch.no_grad():
            y = self.layer(halved)
        self.assertEqual(y.features.dtype, torch.float32)

    def test_matches_plain_spconv(self) -> None:
        plain = spconv.SubMConv3d(32, 32, kernel_size=3, indice_key="k").to(self.device)
        plain.load_state_dict(self.layer.state_dict())
        torch.testing.assert_close(self.layer(self.x).features, plain(self.x).features)


@unittest.skipUnless(T4SPARSE_ON_CUDA, "needs t4sparse.torch and a CUDA device")
class TestT4SparseLayer(unittest.TestCase):
    """The t4sparse flavour dispatches per call and keeps one library per indice_key."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.layer = SubMConv3d(32, 32, kernel_size=3, indice_key="k").to(self.device)
        self.x = make_sparse_tensor(3000, 32, self.device)

    def test_is_the_t4sparse_flavour(self) -> None:
        import t4sparse.torch

        self.assertIs(SubMConv3d, submanifold_conv.T4SparseSubMConv3d)
        self.assertIsInstance(self.layer, t4sparse.torch.SubMConv3d)

    def test_autocast_runs_on_t4sparse_with_fp32_gradients(self) -> None:
        import t4sparse.torch

        with torch.autocast("cuda", dtype=torch.float16):
            y = self.layer(self.x)
        self.assertEqual(y.features.dtype, torch.float16)
        self.assertIsInstance(y.indice_dict["k"], t4sparse.torch.Rulebook)
        y.features.float().square().sum().backward()
        self.assertEqual(self.layer.weight.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(self.layer.weight.grad).all())

    def test_half_features_outside_autocast_run_on_t4sparse(self) -> None:
        import t4sparse.torch

        layer = SubMConv3d(32, 32, kernel_size=3, indice_key="k").to(self.device).half()
        y = layer(self.x.replace_feature(self.x.features.half()))
        self.assertEqual(y.features.dtype, torch.float16)
        self.assertIsInstance(y.indice_dict["k"], t4sparse.torch.Rulebook)

    def test_fp32_outside_autocast_runs_on_spconv_without_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y = self.layer(self.x)
        self.assertEqual(y.features.dtype, torch.float32)
        self.assertNotIsInstance(y.indice_dict["k"], type(None))
        self.assertFalse(
            SubMConv3d.runs_on_t4sparse(self.x.features), "fp32 outside autocast is spconv's"
        )
        twin = self.layer._spconv()
        self.assertIsInstance(twin, spconv.SubMConv3d)
        self.assertIsInstance(y.indice_dict["k"], type(twin(self.x).indice_dict["k"]))

    def test_cpu_tensors_never_run_on_t4sparse(self) -> None:
        for dtype in (torch.float32, torch.float16):
            self.assertFalse(SubMConv3d.runs_on_t4sparse(torch.zeros(1, 32, dtype=dtype)))

    def test_the_two_paths_agree(self) -> None:
        reference = self.layer(self.x).features
        with torch.autocast("cuda", dtype=torch.float16):
            candidate = self.layer(make_sparse_tensor(3000, 32, self.device)).features
        torch.testing.assert_close(candidate.float(), reference, rtol=2e-2, atol=2e-2)

    def test_twin_shares_parameters_and_stays_out_of_the_state_dict(self) -> None:
        twin = self.layer._spconv()
        self.assertIs(twin.weight, self.layer.weight)
        self.assertIs(twin.bias, self.layer.bias)
        self.assertEqual(set(self.layer.state_dict()), {"weight", "bias"})
        self.assertEqual(dict(self.layer.named_modules()), {"": self.layer})

    def test_twin_follows_a_replaced_parameter(self) -> None:
        before = self.layer._spconv()
        self.layer.weight = torch.nn.Parameter(self.layer.weight.detach().clone())
        after = self.layer._spconv()
        self.assertIsNot(before, after)
        self.assertIs(after.weight, self.layer.weight)

    def test_a_key_filled_under_autocast_refuses_an_fp32_call(self) -> None:
        with torch.autocast("cuda", dtype=torch.float16):
            y = self.layer(self.x)
        with self.assertRaisesRegex(TypeError, "indice_key 'k' holds a rulebook"):
            self.layer(y.replace_feature(y.features.float()))

    def test_a_key_filled_by_spconv_refuses_an_autocast_call(self) -> None:
        y = self.layer(self.x)
        with (
            self.assertRaisesRegex(TypeError, "indice_key 'k' holds a"),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            self.layer(y)

    def test_loads_a_plain_spconv_checkpoint(self) -> None:
        plain = spconv.SubMConv3d(32, 32, kernel_size=3, indice_key="k").to(self.device)
        self.layer.load_state_dict(plain.state_dict())
        torch.testing.assert_close(self.layer.weight, plain.weight)


class TestModelIntegration(unittest.TestCase):
    """PTv3 builds its convolutions from this module and still exports through spconv."""

    def test_block_builds_the_layer_from_this_module(self) -> None:
        block = conv_only_block(32)
        self.assertIsInstance(cpe_conv(block), SubMConv3d)
        self.assertTrue(is_sparse_conv_module(cpe_conv(block)))

    def test_export_replacement_converts_the_layer(self) -> None:
        block = conv_only_block(32)
        weight = cpe_conv(block).weight.detach().clone()
        replace_submconv3d_for_export(block)
        self.assertIsInstance(cpe_conv(block), ExportableSubMConv3d)
        self.assertTrue(is_sparse_conv_module(cpe_conv(block)))
        self.assertTrue(torch.equal(cpe_conv(block).weight.detach(), weight))
        self.assertFalse(any(isinstance(module, SubMConv3d) for module in block.modules()))

    @unittest.skipUnless(T4SPARSE_ON_CUDA, "needs t4sparse.torch and a CUDA device")
    def test_t4sparse_block_matches_the_spconv_flavour_under_autocast(self) -> None:
        device = torch.device("cuda")
        t4sparse_block = conv_only_block(32).to(device)
        spconv_block = conv_only_block(32).to(device)
        spconv_block.cpe._modules["0"] = SpconvSubMConv3d(
            32, 32, kernel_size=3, bias=True, indice_key="stage0"
        ).to(device)
        spconv_block.load_state_dict(t4sparse_block.state_dict())

        reference_point = make_point(3000, 32, device)
        candidate_point = make_point(3000, 32, device)
        with torch.autocast("cuda", dtype=torch.float16):
            reference = spconv_block(reference_point).feat
            candidate = t4sparse_block(candidate_point).feat
        torch.testing.assert_close(candidate.float(), reference.float(), rtol=2e-2, atol=2e-2)

        reference.float().square().sum().backward()
        candidate.float().square().sum().backward()
        reference_grad = cpe_conv(spconv_block).weight.grad
        candidate_grad = cpe_conv(t4sparse_block).weight.grad
        self.assertEqual(candidate_grad.dtype, torch.float32)
        torch.testing.assert_close(
            candidate_grad,
            reference_grad,
            rtol=5e-2,
            atol=5e-2 * reference_grad.abs().max().item(),
        )


if __name__ == "__main__":
    unittest.main()
