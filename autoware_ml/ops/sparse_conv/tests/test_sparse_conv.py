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

"""Tests for the sparse convolution backends of the PTv3 family."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    Embedding,
    replace_submconv3d_for_export,
)
from autoware_ml.ops.sparse_conv.backend import SparseConvBackend
from autoware_ml.ops.sparse_conv.rulebook.availability import IS_RULEBOOK_AVAILABLE
from autoware_ml.ops.sparse_conv.tensor import sparse_conv_tensor
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.point_cloud.structures import Point

_BACKEND_MODULE = "autoware_ml.ops.sparse_conv.backend"
_HAS_CUDA = torch.cuda.is_available()


def _make_point(num_points: int, channels: int, device: torch.device, seed: int = 0) -> Point:
    """Build a sparsified two sample point container on a 64^3 grid.

    Args:
        num_points: Number of random grid cells drawn before deduplication.
        channels: Feature width.
        device: Device of the container.
        seed: Seed of the random grid cells and features.

    Returns:
        The point container with its sparse tensor populated.
    """
    generator = torch.Generator().manual_seed(seed)
    grid_coord = torch.randint(0, 64, (num_points, 3), generator=generator)
    grid_coord = torch.unique(grid_coord, dim=0)
    count = grid_coord.shape[0]
    split = count // 2
    feat = torch.randn(count, channels, generator=generator)
    point = Point(
        feat=feat.to(device),
        coord=grid_coord.float().to(device),
        grid_coord=grid_coord.to(device),
        offset=torch.tensor([split, count], dtype=torch.long, device=device),
    )
    point.sparsify()
    return point


def _random_sparse_input(
    num_points: int, channels: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw single sample CUDA features and indices on a 48^3 grid.

    Args:
        num_points: Upper bound of the number of voxels.
        channels: Feature width.
        seed: Seed of the random voxels and features.

    Returns:
        Features ``[N, channels]`` and integer indices ``[N, 4]``.
    """
    generator = torch.Generator().manual_seed(seed)
    coords = torch.randint(0, 48, (num_points * 2, 3), generator=generator)
    coords = torch.unique(coords, dim=0)[:num_points]
    indices = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords.int()], dim=1)
    features = torch.randn(coords.shape[0], channels, generator=generator)
    return features.cuda(), indices.cuda()


def _conv_only_block(channels: int, backend: SparseConvBackend | str) -> Block:
    """Build a convolution-only block, the shape of LitePT's early stages.

    Args:
        channels: Feature width of the block.
        backend: Sparse convolution backend of the positional encoding.

    Returns:
        The block.
    """
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
        sparse_conv_backend=backend,
    )


def _cpe_conv(block: Block) -> torch.nn.Module:
    """Return the sparse convolution at the front of a block's positional encoding.

    Args:
        block: Block carrying a positional encoding.

    Returns:
        The convolution module.
    """
    return block.cpe._modules["0"]


class TestSparseConvBackendResolve(unittest.TestCase):
    """Unit tests for SparseConvBackend.resolve."""

    def _patch(self, *, spconv: bool, rulebook: bool, cuda: bool) -> None:
        """Pretend a set of installed libraries and CUDA availability for one test.

        Args:
            spconv: Whether spconv is installed.
            rulebook: Whether rulebook_torch is installed.
            cuda: Whether CUDA is available.
        """
        for patcher in (
            patch(f"{_BACKEND_MODULE}.IS_SPCONV_AVAILABLE", spconv),
            patch(f"{_BACKEND_MODULE}.IS_RULEBOOK_AVAILABLE", rulebook),
            patch("torch.cuda.is_available", return_value=cuda),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_auto_takes_rulebook_when_installed_and_cuda_is_available(self) -> None:
        """Test that AUTO resolves to RULEBOOK with rulebook installed and CUDA."""
        self._patch(spconv=True, rulebook=True, cuda=True)
        self.assertIs(SparseConvBackend.AUTO.resolve(), SparseConvBackend.RULEBOOK)

    def test_auto_falls_back_to_spconv(self) -> None:
        """Test that AUTO resolves to SPCONV without rulebook or without CUDA."""
        for rulebook, cuda in ((False, True), (True, False), (False, False)):
            with self.subTest(rulebook=rulebook, cuda=cuda):
                self._patch(spconv=True, rulebook=rulebook, cuda=cuda)
                self.assertIs(SparseConvBackend.AUTO.resolve(), SparseConvBackend.SPCONV)

    def test_explicit_backends_resolve_to_themselves(self) -> None:
        """Test that SPCONV and RULEBOOK resolve to themselves, CUDA or not."""
        self._patch(spconv=True, rulebook=True, cuda=False)
        for backend in (SparseConvBackend.SPCONV, SparseConvBackend.RULEBOOK):
            with self.subTest(backend=backend):
                self.assertIs(backend.resolve(), backend)

    def test_resolve_rejects_missing_libraries(self) -> None:
        """Test that a backend whose library is missing raises ModuleNotFoundError."""
        cases = (
            (SparseConvBackend.AUTO, False, True, False, "'auto' needs 'spconv'"),
            (SparseConvBackend.RULEBOOK, True, False, True, "'rulebook_torch'"),
            (SparseConvBackend.SPCONV, False, True, True, "'spconv'"),
        )
        for backend, spconv, rulebook, cuda, message in cases:
            with self.subTest(backend=backend):
                self._patch(spconv=spconv, rulebook=rulebook, cuda=cuda)
                with self.assertRaisesRegex(ModuleNotFoundError, message):
                    backend.resolve()

    def test_unknown_name_is_rejected(self) -> None:
        """Test that an unknown backend name raises ValueError."""
        with self.assertRaisesRegex(ValueError, "torchsparse"):
            SparseConvBackend("torchsparse")

    def test_names_match_the_config_values(self) -> None:
        """Test that the members carry the names the configs pass."""
        self.assertEqual(
            [str(backend) for backend in SparseConvBackend], ["auto", "spconv", "rulebook"]
        )


class TestSparseConvBackendType(unittest.TestCase):
    """Unit tests for the SparseConvBackend type checks of the PTv3 layers."""

    def test_block_rejects_a_plain_string(self) -> None:
        """Test that Block raises TypeError for a backend given as a string."""
        with self.assertRaisesRegex(TypeError, "must be a SparseConvBackend, got str"):
            _conv_only_block(32, "spconv")

    def test_embedding_rejects_a_plain_string(self) -> None:
        """Test that Embedding raises TypeError for a backend given as a string."""
        with self.assertRaisesRegex(TypeError, "must be a SparseConvBackend, got str"):
            Embedding(4, 32, stem_type="linear", sparse_conv_backend="spconv")


@unittest.skipUnless(IS_SPCONV_AVAILABLE, "spconv is not installed")
class TestSpconvInspection(unittest.TestCase):
    """Unit tests for the sparse tensor and module inspection on spconv."""

    def test_sparsify_builds_a_sparse_tensor_the_backends_accept(self) -> None:
        """Test that Point.sparsify builds a sparse tensor with the shared interface."""
        sparse_feat = _make_point(200, 8, torch.device("cpu")).sparse_conv_feat

        self.assertTrue(SparseConvBackend.is_tensor(sparse_feat))
        self.assertEqual(sparse_feat.features.shape[1], 8)
        self.assertEqual(sparse_feat.indices.shape[1], 4)
        self.assertEqual(sparse_feat.batch_size, 2)
        self.assertIsInstance(sparse_feat.indice_dict, dict)
        replaced = sparse_feat.replace_feature(sparse_feat.features * 2)
        self.assertTrue(torch.equal(replaced.features, sparse_feat.features * 2))

    def test_dense_values_are_not_sparse_tensors(self) -> None:
        """Test that dense tensors are not taken for sparse tensors."""
        self.assertFalse(SparseConvBackend.is_tensor(torch.zeros(3, 4)))

    def test_of_module(self) -> None:
        """Test the backend reported for submanifold, strided and dense modules."""
        import spconv.pytorch as spconv

        submanifold = SparseConvBackend.SPCONV.submanifold_conv3d(
            8, 8, 3, bias=True, indice_key="k"
        )
        strided = spconv.SparseConv3d(8, 8, 3, stride=2)
        cases = (
            (submanifold, False, SparseConvBackend.SPCONV),
            (submanifold, True, SparseConvBackend.SPCONV),
            (strided, False, SparseConvBackend.SPCONV),
            (strided, True, None),
            (torch.nn.Linear(8, 8), False, None),
        )
        for module, submanifold_only, expected in cases:
            with self.subTest(module=type(module).__name__, submanifold_only=submanifold_only):
                self.assertIs(
                    SparseConvBackend.of_module(module, submanifold_only=submanifold_only),
                    expected,
                )

    def test_is_module(self) -> None:
        """Test that every spconv module, strided ones included, counts as sparse."""
        import spconv.pytorch as spconv

        cases = (
            (SparseConvBackend.SPCONV.submanifold_conv3d(8, 8, 3, bias=True, indice_key="k"), True),
            (spconv.SparseConv3d(8, 8, 3, stride=2), True),
            (torch.nn.Linear(8, 8), False),
        )
        for module, expected in cases:
            with self.subTest(module=type(module).__name__):
                self.assertEqual(SparseConvBackend.is_module(module), expected)


@unittest.skipUnless(IS_RULEBOOK_AVAILABLE, "rulebook_torch is not installed")
class TestRulebookSparseTensor(unittest.TestCase):
    """Unit tests for the sparse tensor container without spconv."""

    def test_sparse_conv_tensor_without_spconv_uses_the_rulebook_container(self) -> None:
        """Test that the rulebook container is built when spconv is missing."""
        import rulebook_torch

        features = torch.zeros(3, 4)
        indices = torch.tensor([[0, 1, 2, 3], [0, 2, 2, 3], [1, 0, 0, 0]], dtype=torch.int32)
        with patch("autoware_ml.ops.sparse_conv.tensor.IS_SPCONV_AVAILABLE", False):
            tensor = sparse_conv_tensor(features, indices, [8, 8, 8], 2)

        self.assertIsInstance(tensor, rulebook_torch.SparseTensor)
        self.assertTrue(SparseConvBackend.is_tensor(tensor))


@unittest.skipUnless(
    IS_SPCONV_AVAILABLE and IS_RULEBOOK_AVAILABLE and _HAS_CUDA,
    "comparing the backends needs spconv, rulebook_torch and a CUDA device",
)
class TestRulebookAgainstSpconv(unittest.TestCase):
    """Unit tests comparing rulebook blocks with spconv blocks."""

    def test_rulebook_block_matches_spconv_block_under_fp16_autocast(self) -> None:
        """Test that both backends agree in forward and weight gradient."""
        device = torch.device("cuda")
        channels = 32
        spconv_block = _conv_only_block(channels, SparseConvBackend.SPCONV).to(device)
        rulebook_block = _conv_only_block(channels, SparseConvBackend.RULEBOOK).to(device)
        self.assertIs(
            SparseConvBackend.of_module(_cpe_conv(spconv_block), submanifold_only=True),
            SparseConvBackend.SPCONV,
        )
        self.assertIs(
            SparseConvBackend.of_module(_cpe_conv(rulebook_block), submanifold_only=True),
            SparseConvBackend.RULEBOOK,
        )
        # Same parameter names and shapes: a checkpoint of one backend loads on the other.
        rulebook_block.load_state_dict(spconv_block.state_dict())

        reference_point = _make_point(3000, channels, device)
        candidate_point = _make_point(3000, channels, device)
        reference_point.feat.requires_grad_(True)
        candidate_point.feat.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            reference = spconv_block(reference_point).feat
            candidate = rulebook_block(candidate_point).feat
        torch.testing.assert_close(candidate.float(), reference.float(), rtol=2e-2, atol=2e-2)

        reference.float().square().sum().backward()
        candidate.float().square().sum().backward()
        spconv_grad = _cpe_conv(spconv_block).weight.grad
        rulebook_grad = _cpe_conv(rulebook_block).weight.grad
        torch.testing.assert_close(
            rulebook_grad, spconv_grad, rtol=5e-2, atol=5e-2 * spconv_grad.abs().max().item()
        )
        self.assertEqual(rulebook_grad.dtype, torch.float32)

    def test_export_replacement_converts_rulebook_layers(self) -> None:
        """Test that export replaces rulebook layers with the spconv export wrapper."""
        from autoware_ml.ops.spconv.sparse_conv import SubMConv3d as ExportableSubMConv3d

        block = _conv_only_block(32, SparseConvBackend.RULEBOOK).to(torch.device("cuda"))
        weight = _cpe_conv(block).weight.detach().clone()
        replace_submconv3d_for_export(block)

        self.assertIsInstance(_cpe_conv(block), ExportableSubMConv3d)
        # The export wrapper derives from spconv's base convolution, not from SubMConv3d.
        self.assertIs(SparseConvBackend.of_module(_cpe_conv(block)), SparseConvBackend.SPCONV)
        self.assertTrue(torch.equal(_cpe_conv(block).weight.detach(), weight))
        self.assertTrue(
            all(
                SparseConvBackend.of_module(module, submanifold_only=True)
                is not SparseConvBackend.RULEBOOK
                for module in block.modules()
            )
        )


@unittest.skipUnless(IS_RULEBOOK_AVAILABLE and _HAS_CUDA, "rulebook_torch on CUDA required")
class TestPaddedSubMConv3d(unittest.TestCase):
    """Unit tests for the channel padding of the rulebook layer."""

    def test_padded_widths_are_compiled_and_cover_the_true_widths(self) -> None:
        """Test that padded widths are compiled and at least the true widths."""
        from rulebook_torch._ops import has_kernel

        from autoware_ml.ops.sparse_conv.rulebook.padded_conv import PaddedSubMConv3d

        kernel_size = (3, 3, 3)
        for c_in, c_out in ((54, 54), (108, 108), (216, 216), (432, 432), (30, 50)):
            with self.subTest(c_in=c_in, c_out=c_out):
                widths = PaddedSubMConv3d.padded_widths(c_in, c_out, kernel_size, torch.float16)
                if PaddedSubMConv3d.has_kernels(c_in, c_out, kernel_size, torch.float16):
                    self.assertIsNone(widths)
                    continue
                self.assertIsNotNone(widths)
                self.assertGreaterEqual(widths.in_channels, c_in)
                self.assertGreaterEqual(widths.out_channels, c_out)
                self.assertTrue(PaddedSubMConv3d.has_kernels(*widths, kernel_size, torch.float16))
        # A width with its own kernels is left alone.
        if has_kernel(64, 64, kernel_size, torch.float16):
            self.assertIsNone(PaddedSubMConv3d.padded_widths(64, 64, kernel_size, torch.float16))

    def test_padded_layer_matches_the_reference_forward_and_backward(self) -> None:
        """Test that a padded layer reproduces the fp32 reference convolution."""
        from rulebook_torch._ops import conv_reference, rulebook_for

        from autoware_ml.ops.sparse_conv.rulebook.padded_conv import PaddedSubMConv3d

        for c_in, c_out in ((54, 54), (30, 50)):
            with self.subTest(c_in=c_in, c_out=c_out):
                torch.manual_seed(0)
                layer = SparseConvBackend.RULEBOOK.submanifold_conv3d(
                    c_in, c_out, 3, bias=True, indice_key="stage0"
                ).cuda()
                self.assertIsInstance(layer, PaddedSubMConv3d)
                self.assertIsNotNone(
                    PaddedSubMConv3d.padded_widths(c_in, c_out, (3, 3, 3), torch.float16)
                )

                features, indices = _random_sparse_input(4000, c_in, seed=1)
                features.requires_grad_(True)
                x = sparse_conv_tensor(features, indices, [48, 48, 48], 1)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = layer(x).features
                self.assertEqual(tuple(out.shape), (features.shape[0], c_out))
                self.assertEqual(out.dtype, torch.float16)

                # fp32 reference on the same rulebook, same weights.
                rb = rulebook_for(x, None, 3)
                ref_features = features.detach().clone().requires_grad_(True)
                ref_weight = layer.weight.detach().clone().requires_grad_(True)
                ref = conv_reference(ref_features, ref_weight, rb) + layer.bias.detach()
                torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)

                probe = torch.randn_like(ref)
                (out.float() * probe).sum().backward()
                (ref * probe).sum().backward()
                torch.testing.assert_close(features.grad, ref_features.grad, atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(layer.weight.grad, ref_weight.grad, atol=5e-2, rtol=5e-2)
                self.assertEqual(tuple(layer.weight.grad.shape), (c_out, 3, 3, 3, c_in))
                self.assertIsNotNone(layer.bias.grad)
                self.assertTrue(torch.isfinite(layer.bias.grad).all())

    def test_padded_layer_state_dict_matches_the_plain_layer(self) -> None:
        """Test that the padded layer's state dict loads into the plain layer."""
        import rulebook_torch

        layer = SparseConvBackend.RULEBOOK.submanifold_conv3d(54, 54, 3, bias=True, indice_key="k")
        plain = rulebook_torch.SubMConv3d(54, 54, kernel_size=3, bias=True, indice_key="k")
        plain.load_state_dict(layer.state_dict())

        self.assertTrue(torch.equal(plain.weight, layer.weight))

    def test_fp32_features_outside_autocast_warn_once(self) -> None:
        """Test that fp32 features outside autocast log one warning per process."""
        from autoware_ml.ops.sparse_conv.rulebook.padded_conv import PaddedSubMConv3d

        layer = SparseConvBackend.RULEBOOK.submanifold_conv3d(
            32, 32, 3, bias=True, indice_key="k"
        ).cuda()
        features, indices = _random_sparse_input(100, 32, seed=2)
        with (
            patch.object(PaddedSubMConv3d, "_fp32_warned", False),
            self.assertLogs("autoware_ml.ops.sparse_conv.rulebook.padded_conv", "WARNING") as logs,
        ):
            layer(sparse_conv_tensor(features, indices, [48, 48, 48], 1))
            layer(sparse_conv_tensor(features, indices, [48, 48, 48], 1))

        # The width search may log its own warning for the missing fp32 kernel; only the
        # fallback warning is limited to one per process.
        fp32_warnings = [line for line in logs.output if "fp32 features outside autocast" in line]
        self.assertEqual(len(fp32_warnings), 1)
