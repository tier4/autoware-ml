"""Tests for the sparse-voxel BEVFusion lidar encoders."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.encoders.voxel import HardSimpleVoxelSinCosEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

_MIN = [-122.4, -122.4, -3.0, 0.0]
_MAX = [122.4, 122.4, 5.0, 255.0]


def test_voxel_encoder_output_dim_and_sincos() -> None:
    encoder = HardSimpleVoxelSinCosEncoder(_MIN, _MAX, in_channels=4)
    voxels = torch.randn(9, 32, 4)
    num_points = torch.randint(1, 32, (9,))
    out = encoder(voxels, num_points, coords=torch.zeros(9, 4))
    # Output is 2 * C**2 (cos and sin of the C*C Fourier features).
    assert out.shape == (9, 2 * 4 * 4)
    # cos(x)**2 + sin(x)**2 == 1 for every Fourier component.
    cos_part, sin_part = out[:, : 4 * 4], out[:, 4 * 4 :]
    assert torch.allclose(cos_part**2 + sin_part**2, torch.ones_like(cos_part), atol=1e-5)


def test_voxel_encoder_mean_pooling_ignores_padding() -> None:
    encoder = HardSimpleVoxelSinCosEncoder(_MIN, _MAX, in_channels=4)
    voxels = torch.zeros(1, 4, 4)
    voxels[0, 0] = torch.tensor([10.0, 20.0, 1.0, 100.0])  # one real point, rest padding
    out_one = encoder(voxels, torch.tensor([1]), coords=torch.zeros(1, 4))
    # Same single real point with explicit 1-point count  identical features.
    single = torch.tensor([[[10.0, 20.0, 1.0, 100.0]]])
    out_single = encoder(single, torch.tensor([1]), coords=torch.zeros(1, 4))
    assert torch.allclose(out_one, out_single, atol=1e-5)


@pytest.mark.skipif(not IS_SPCONV_AVAILABLE, reason="SparseEncoder requires spconv")
def test_sparse_encoder_prepare_for_export_replaces_sparse_convolutions() -> None:
    from autoware_ml.models.detection3d.encoders.sparse import SparseConv3d as NativeSparseConv3d
    from autoware_ml.models.detection3d.encoders.sparse import SparseEncoder
    from autoware_ml.models.detection3d.encoders.sparse import SubMConv3d as NativeSubMConv3d
    from autoware_ml.ops.spconv.sparse_conv import SparseConv3d as ExportableSparseConv3d
    from autoware_ml.ops.spconv.sparse_conv import SubMConv3d as ExportableSubMConv3d

    encoder = SparseEncoder(
        in_channels=32,
        sparse_shape=[16, 16, 5],
        output_channels=16,
        dense_output_shapes=[2, 2, 1],
    )

    export_encoder = encoder.prepare_for_export()

    assert export_encoder is not encoder
    assert isinstance(encoder.conv_input[0], NativeSubMConv3d)
    assert isinstance(export_encoder.conv_input[0], ExportableSubMConv3d)
    assert not any(
        isinstance(module, (NativeSubMConv3d, NativeSparseConv3d))
        for module in export_encoder.modules()
    )
    assert any(isinstance(module, ExportableSparseConv3d) for module in export_encoder.modules())
    assert torch.equal(export_encoder.conv_input[0].weight, encoder.conv_input[0].weight)

    export_encoder.conv_input[0].weight.data.add_(1.0)
    assert not torch.equal(export_encoder.conv_input[0].weight, encoder.conv_input[0].weight)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="BEVFusion sparse encoder requires CUDA spconv",
)
def test_sparse_encoder_produces_dense_bev() -> None:
    from autoware_ml.models.detection3d.encoders.sparse import SparseEncoder

    device = torch.device("cuda")
    encoder = SparseEncoder(
        in_channels=32,
        sparse_shape=[1440, 1440, 41],
        output_channels=128,
        dense_output_shapes=[180, 180, 2],
    ).to(device)

    num_voxels, batch_size = 3000, 2
    features = torch.randn(num_voxels, 32, device=device, requires_grad=True)
    coords = torch.zeros(num_voxels, 4, dtype=torch.int32, device=device)
    coords[:, 0] = torch.randint(0, batch_size, (num_voxels,), device=device)  # batch
    coords[:, 1] = torch.randint(0, 41, (num_voxels,), device=device)  # z
    coords[:, 2] = torch.randint(0, 1440, (num_voxels,), device=device)  # y
    coords[:, 3] = torch.randint(0, 1440, (num_voxels,), device=device)  # x

    bev = encoder(features, coords, batch_size)
    # output_channels * Z, grid / out_size_factor (1440 / 8 = 180)
    assert bev.shape == (batch_size, 128 * 2, 180, 180)
    assert torch.isfinite(bev).all()
    bev.sum().backward()
    assert features.grad is not None
