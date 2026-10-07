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

"""Tests for the sparse convolution backend selection of the PTv3 family."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    replace_submconv3d_for_export,
)
from autoware_ml.ops import sparse_backend
from autoware_ml.utils.point_cloud.structures import Point

requires_spconv = pytest.mark.skipif(
    not sparse_backend.IS_SPCONV_AVAILABLE, reason="spconv is not installed"
)
requires_rulebook = pytest.mark.skipif(
    not sparse_backend.IS_RULEBOOK_AVAILABLE, reason="rulebook_torch is not installed"
)
requires_both_on_cuda = pytest.mark.skipif(
    not (
        sparse_backend.IS_SPCONV_AVAILABLE
        and sparse_backend.IS_RULEBOOK_AVAILABLE
        and torch.cuda.is_available()
    ),
    reason="comparing the backends needs spconv, rulebook_torch and a CUDA device",
)


def test_auto_prefers_rulebook_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sparse_backend, "IS_SPCONV_AVAILABLE", True)
    monkeypatch.setattr(sparse_backend, "IS_RULEBOOK_AVAILABLE", True)
    assert sparse_backend.resolve_sparse_conv_backend("auto") == "rulebook"
    assert sparse_backend.resolve_sparse_conv_backend("spconv") == "spconv"


def test_auto_falls_back_to_spconv_without_rulebook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sparse_backend, "IS_SPCONV_AVAILABLE", True)
    monkeypatch.setattr(sparse_backend, "IS_RULEBOOK_AVAILABLE", False)
    assert sparse_backend.resolve_sparse_conv_backend("auto") == "spconv"


def test_resolve_rejects_missing_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sparse_backend, "IS_SPCONV_AVAILABLE", False)
    monkeypatch.setattr(sparse_backend, "IS_RULEBOOK_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match="Neither"):
        sparse_backend.resolve_sparse_conv_backend("auto")
    monkeypatch.setattr(sparse_backend, "IS_SPCONV_AVAILABLE", True)
    with pytest.raises(ModuleNotFoundError, match="rulebook_torch"):
        sparse_backend.resolve_sparse_conv_backend("rulebook")
    monkeypatch.setattr(sparse_backend, "IS_SPCONV_AVAILABLE", False)
    monkeypatch.setattr(sparse_backend, "IS_RULEBOOK_AVAILABLE", True)
    with pytest.raises(ModuleNotFoundError, match="spconv"):
        sparse_backend.resolve_sparse_conv_backend("spconv")


def test_resolve_rejects_unknown_backend_name() -> None:
    with pytest.raises(ValueError, match="Unknown sparse convolution backend"):
        sparse_backend.resolve_sparse_conv_backend("torchsparse")


def _make_point(num_points: int, channels: int, device: torch.device, seed: int = 0) -> Point:
    """Return a two sample point container with voxel features on a 64^3 grid."""
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


@requires_spconv
def test_sparsify_builds_a_sparse_tensor_the_backends_accept() -> None:
    point = _make_point(200, 8, torch.device("cpu"))
    sparse_feat = point.sparse_conv_feat
    assert sparse_backend.is_sparse_conv_tensor(sparse_feat)
    assert sparse_feat.features.shape[1] == 8
    assert sparse_feat.indices.shape[1] == 4
    assert sparse_feat.batch_size == 2
    assert isinstance(sparse_feat.indice_dict, dict)
    replaced = sparse_feat.replace_feature(sparse_feat.features * 2)
    assert torch.equal(replaced.features, sparse_feat.features * 2)


@requires_rulebook
def test_sparse_conv_tensor_without_spconv_uses_the_rulebook_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rulebook_torch

    monkeypatch.setattr(sparse_backend, "IS_SPCONV_AVAILABLE", False)
    features = torch.zeros(3, 4)
    indices = torch.tensor([[0, 1, 2, 3], [0, 2, 2, 3], [1, 0, 0, 0]], dtype=torch.int32)
    tensor = sparse_backend.sparse_conv_tensor(features, indices, [8, 8, 8], 2)
    assert isinstance(tensor, rulebook_torch.SparseTensor)
    assert sparse_backend.is_sparse_conv_tensor(tensor)


def _cpe_conv(block: Block) -> torch.nn.Module:
    """Return the sparse convolution at the front of a block's positional encoding."""
    return block.cpe._modules["0"]


def _conv_only_block(channels: int, backend: str) -> Block:
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


@requires_both_on_cuda
def test_rulebook_block_matches_spconv_block_under_fp16_autocast() -> None:
    device = torch.device("cuda")
    channels = 32
    spconv_block = _conv_only_block(channels, "spconv").to(device)
    rulebook_block = _conv_only_block(channels, "rulebook").to(device)
    assert sparse_backend.submanifold_conv_backend_of(_cpe_conv(spconv_block)) == "spconv"
    assert sparse_backend.submanifold_conv_backend_of(_cpe_conv(rulebook_block)) == "rulebook"
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
    torch.testing.assert_close(
        _cpe_conv(rulebook_block).weight.grad,
        _cpe_conv(spconv_block).weight.grad,
        rtol=5e-2,
        atol=5e-2 * _cpe_conv(spconv_block).weight.grad.abs().max().item(),
    )
    assert _cpe_conv(rulebook_block).weight.grad.dtype == torch.float32


@requires_both_on_cuda
def test_export_replacement_converts_rulebook_layers() -> None:
    from autoware_ml.ops.spconv.sparse_conv import SubMConv3d as ExportableSubMConv3d

    block = _conv_only_block(32, "rulebook").to(torch.device("cuda"))
    weight = _cpe_conv(block).weight.detach().clone()
    replace_submconv3d_for_export(block)
    assert isinstance(_cpe_conv(block), ExportableSubMConv3d)
    # The export wrapper derives from spconv's base convolution, not from SubMConv3d.
    assert sparse_backend.sparse_conv_backend_of(_cpe_conv(block)) == "spconv"
    assert sparse_backend.is_sparse_conv_module(_cpe_conv(block))
    assert torch.equal(_cpe_conv(block).weight.detach(), weight)
    assert all(
        sparse_backend.submanifold_conv_backend_of(module) != "rulebook"
        for module in block.modules()
    )


requires_rulebook_cuda = pytest.mark.skipif(
    not (sparse_backend.IS_RULEBOOK_AVAILABLE and torch.cuda.is_available()),
    reason="rulebook_torch on CUDA required",
)


def _random_sparse_tensor(num_points: int, channels: int, seed: int) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    coords = torch.randint(0, 48, (num_points * 2, 3), generator=generator)
    coords = torch.unique(coords, dim=0)[:num_points]
    indices = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords.int()], dim=1)
    features = torch.randn(coords.shape[0], channels, generator=generator)
    return features.cuda(), indices.cuda()


@requires_rulebook_cuda
def test_padded_widths_are_compiled_and_cover_the_true_widths() -> None:
    from rulebook_torch._ops import has_kernel, has_weight_grad_kernel

    for c_in, c_out in [(54, 54), (108, 108), (216, 216), (432, 432), (30, 50)]:
        widths = sparse_backend.rulebook_padded_widths(c_in, c_out, (3, 3, 3), torch.float16)
        if has_kernel(c_in, c_out, (3, 3, 3), torch.float16) and has_weight_grad_kernel(
            c_in, c_out, torch.float16
        ):
            assert widths is None
            continue
        assert widths is not None, (c_in, c_out)
        padded_in, padded_out = widths
        assert padded_in >= c_in and padded_out >= c_out
        assert has_kernel(padded_in, padded_out, (3, 3, 3), torch.float16)
        assert has_weight_grad_kernel(padded_in, padded_out, torch.float16)
    # A width with its own kernels is left alone.
    if has_kernel(64, 64, (3, 3, 3), torch.float16):
        assert sparse_backend.rulebook_padded_widths(64, 64, (3, 3, 3), torch.float16) is None


@requires_rulebook_cuda
@pytest.mark.parametrize("c_in,c_out", [(54, 54), (30, 50)])
def test_padded_rulebook_layer_matches_the_reference_forward_and_backward(
    c_in: int, c_out: int
) -> None:
    import rulebook_torch
    from rulebook_torch._ops import conv_reference, rulebook_for

    torch.manual_seed(0)
    layer = sparse_backend.submanifold_conv3d(
        c_in, c_out, 3, bias=True, indice_key="stage0", backend="rulebook"
    ).cuda()
    assert isinstance(layer, rulebook_torch.SubMConv3d)
    assert type(layer).__name__ == "PaddedSubMConv3d"
    assert sparse_backend.rulebook_padded_widths(c_in, c_out, (3, 3, 3), torch.float16) is not None

    features, indices = _random_sparse_tensor(4000, c_in, seed=1)
    features.requires_grad_(True)
    x = sparse_backend.sparse_conv_tensor(features, indices, [48, 48, 48], 1)
    with torch.autocast("cuda", dtype=torch.float16):
        out = layer(x).features
    assert out.shape == (features.shape[0], c_out)
    assert out.dtype == torch.float16

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
    assert layer.weight.grad.shape == (c_out, 3, 3, 3, c_in)
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()


@requires_rulebook_cuda
def test_padded_layer_state_dict_matches_the_plain_layer() -> None:
    import rulebook_torch

    layer = sparse_backend.submanifold_conv3d(
        54, 54, 3, bias=True, indice_key="k", backend="rulebook"
    )
    plain = rulebook_torch.SubMConv3d(54, 54, kernel_size=3, bias=True, indice_key="k")
    plain.load_state_dict(layer.state_dict())
    assert torch.equal(plain.weight, layer.weight)
