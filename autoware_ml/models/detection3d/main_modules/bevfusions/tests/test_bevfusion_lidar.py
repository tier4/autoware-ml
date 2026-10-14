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

"""Unit tests for the BEVFusion lidar branch."""

from __future__ import annotations

from collections.abc import Sequence
import unittest

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_lidar import BEVFusionLidar
from autoware_ml.models.detection3d.main_modules.bevfusions.fuser import ConvFuser
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN


class _StubVoxelEncoder(nn.Module):
    """Mean-pool the points of each voxel and project them to the middle encoder channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_points channels"],
        num_points: Int32[torch.Tensor, " num_voxels"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
    ) -> Float32[torch.Tensor, "num_voxels out_channels"]:
        voxel_mean = voxels.sum(dim=1) / num_points.clamp(min=1).unsqueeze(1).to(voxels.dtype)
        return self.linear(voxel_mean)


class _StubMiddleEncoder(nn.Module):
    """Scatter voxel features onto a dense ``(B, C, H, W)`` canvas at their ``(y, x)`` cells.

    Coordinates arrive in the ``(batch, z, y, x)`` layout the main model builds. The stub exposes
    the ``bev_output_shape`` and ``prepare_for_export`` interface of the sparse encoder.
    """

    def __init__(self, bev_shape: tuple[int, int], exportable: bool = False) -> None:
        super().__init__()
        self._bev_shape = bev_shape
        self.exportable = exportable

    @property
    def bev_output_shape(self) -> tuple[int, int]:
        return self._bev_shape

    def forward(
        self,
        voxel_features: Float32[torch.Tensor, "num_voxels channels"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
        batch_size: int,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        height, width = self._bev_shape
        canvas = voxel_features.new_zeros(batch_size, voxel_features.shape[1], height, width)
        batch_indices, y, x = coords[:, 0].long(), coords[:, 2].long(), coords[:, 3].long()
        canvas[batch_indices, :, y, x] = voxel_features
        return canvas

    def prepare_for_export(self) -> _StubMiddleEncoder:
        return _StubMiddleEncoder(self._bev_shape, exportable=True)


class _RecordingConvFuser(ConvFuser):
    """ConvFuser that records the shapes of the feature maps it is asked to fuse."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.received_shapes: list[list[tuple[int, ...]]] = []

    def forward(
        self, bev_features: Sequence[Float32[torch.Tensor, "batch_size channels height width"]]
    ) -> Float32[torch.Tensor, "batch_size out_channels height width"]:
        self.received_shapes.append([tuple(features.shape) for features in bev_features])
        return super().forward(bev_features)


class _BEVFusionLidarTestCase(unittest.TestCase):
    """Shared configuration and helpers for the lidar branch test cases.

    Subclasses build the encoders and the voxel batch declared below in their own ``setUp``.
    """

    voxel_encoder: nn.Module
    middle_encoder: nn.Module
    backbone: nn.Module
    neck: nn.Module
    voxels: Float32[torch.Tensor, "num_voxels max_points channels"]
    coords: Int32[torch.Tensor, "num_voxels 4"]
    num_points: Int32[torch.Tensor, " num_voxels"]

    def setUp(self) -> None:
        """Set up the batch layout and device shared by every lidar branch test."""
        torch.manual_seed(0)
        self.batch_size = 2
        self.point_channels = 4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_lidar(self, fuser: ConvFuser | None = None) -> BEVFusionLidar:
        """Build a lidar branch from the shared encoders on the test device in eval mode."""
        return (
            BEVFusionLidar(
                pts_voxel_encoder=self.voxel_encoder,
                pts_middle_encoder=self.middle_encoder,
                pts_backbone=self.backbone,
                pts_neck=self.neck,
                fuser=fuser,
            )
            .to(self.device)
            .eval()
        )

    def _forward(
        self,
        lidar: BEVFusionLidar,
        other_bev_features: Sequence[Float32[torch.Tensor, "batch_size channels height width"]]
        | None,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Run the branch on the shared voxel batch."""
        with torch.no_grad():
            return lidar(
                voxels=self.voxels,
                coords=self.coords,
                num_points=self.num_points,
                batch_size=self.batch_size,
                other_bev_features=other_bev_features,
            )

    def _distinct_cells(
        self, num_voxels_per_sample: int, num_cells: int
    ) -> Int32[torch.Tensor, " num_voxels"]:
        """Draw distinct flat cell indices per sample and concatenate them over the batch."""
        return torch.stack(
            [
                torch.randperm(num_cells, device=self.device)[:num_voxels_per_sample]
                for _ in range(self.batch_size)
            ]
        ).flatten()

    def _batch_indices(self, num_voxels_per_sample: int) -> Int32[torch.Tensor, " num_voxels"]:
        """Build the batch index column for a batch with a fixed voxel count per sample."""
        return torch.arange(self.batch_size, device=self.device).repeat_interleave(
            num_voxels_per_sample
        )


class TestBEVFusionLidar(_BEVFusionLidarTestCase):
    """Unit tests for the lidar branch wiring, run with lightweight stand-in encoders."""

    def setUp(self) -> None:
        """Set up shared encoders and a small two-sample voxel batch."""
        super().setUp()
        self.middle_channels = 16
        self.image_channels = 4
        self.neck_channels = 32
        self.bev_shape = (8, 8)

        self.voxel_encoder = _StubVoxelEncoder(self.point_channels, self.middle_channels).to(
            self.device
        )
        self.middle_encoder = _StubMiddleEncoder(self.bev_shape).to(self.device)
        self.backbone = SECONDBackbone(
            in_channels=self.middle_channels,
            out_channels=[16, 32],
            layer_nums=[1, 1],
            layer_strides=[1, 2],
        ).to(self.device)
        self.neck = SECONDFPN(
            in_channels=[16, 32],
            out_channels=[self.neck_channels // 2, self.neck_channels // 2],
            upsample_strides=[1, 2],
        ).to(self.device)

        num_voxels_per_sample, max_points = 6, 5
        num_voxels = self.batch_size * num_voxels_per_sample
        height, width = self.bev_shape
        self.voxels = torch.randn(num_voxels, max_points, self.point_channels, device=self.device)
        self.num_points = torch.randint(
            1, max_points + 1, (num_voxels,), dtype=torch.int32, device=self.device
        )
        # (batch, z, y, x) coordinates with distinct cells per sample.
        cells = self._distinct_cells(num_voxels_per_sample, height * width)
        self.coords = torch.stack(
            [
                self._batch_indices(num_voxels_per_sample),
                torch.zeros(num_voxels, dtype=torch.int64, device=self.device),
                cells // width,
                cells % width,
            ],
            dim=1,
        ).int()
        self.image_bev: Float32[torch.Tensor, "batch_size channels height width"] = torch.randn(
            self.batch_size, self.image_channels, *self.bev_shape, device=self.device
        )

    def _build_fuser(self) -> _RecordingConvFuser:
        """Build a recording fuser that merges the image BEV into the middle encoder channels."""
        return (
            _RecordingConvFuser(
                in_channels=[self.image_channels, self.middle_channels],
                out_channels=self.middle_channels,
            )
            .to(self.device)
            .eval()
        )

    def test_expected_bev_shape_comes_from_middle_encoder(self) -> None:
        """Test that the branch reports the middle encoder's dense BEV shape."""
        self.assertEqual(self._build_lidar().expected_bev_shape, self.bev_shape)

    def test_expected_bev_shape_requires_bev_output_shape(self) -> None:
        """Test that a middle encoder without ``bev_output_shape`` is rejected."""
        lidar = BEVFusionLidar(
            pts_voxel_encoder=self.voxel_encoder,
            pts_middle_encoder=nn.Identity().to(self.device),
            pts_backbone=self.backbone,
            pts_neck=self.neck,
            fuser=None,
        )

        with self.assertRaises(AttributeError):
            _ = lidar.expected_bev_shape

    def test_forward_without_fusion_returns_neck_features(self) -> None:
        """Test that the lidar-only path produces a neck feature map on the middle encoder grid."""
        bev = self._forward(self._build_lidar(), other_bev_features=None)

        self.assertEqual(bev.shape, (self.batch_size, self.neck_channels, *self.bev_shape))
        self.assertTrue(torch.isfinite(bev).all())

    def test_forward_fuses_other_bev_features_before_backbone(self) -> None:
        """
        Test that the fuser receives the other-modality maps first and the lidar map last, and
        that the fused map flows through the backbone and neck.
        """
        fuser = self._build_fuser()

        bev = self._forward(self._build_lidar(fuser=fuser), other_bev_features=[self.image_bev])

        self.assertEqual(bev.shape, (self.batch_size, self.neck_channels, *self.bev_shape))
        self.assertEqual(
            fuser.received_shapes,
            [
                [
                    (self.batch_size, self.image_channels, *self.bev_shape),
                    (self.batch_size, self.middle_channels, *self.bev_shape),
                ]
            ],
        )
        # Fusion changes the map that reaches the backbone.
        lidar_only = self._forward(self._build_lidar(), other_bev_features=None)
        self.assertFalse(torch.allclose(bev, lidar_only))

    def test_forward_skips_fuser_without_other_bev_features(self) -> None:
        """Test that a configured fuser is bypassed when no other modality is provided."""
        fuser = self._build_fuser()

        bev = self._forward(self._build_lidar(fuser=fuser), other_bev_features=None)

        self.assertEqual(fuser.received_shapes, [])
        torch.testing.assert_close(bev, self._forward(self._build_lidar(), other_bev_features=None))

    def test_forward_ignores_other_bev_features_without_fuser(self) -> None:
        """Test that other-modality maps are ignored when the branch has no fuser."""
        lidar = self._build_lidar()

        torch.testing.assert_close(
            self._forward(lidar, other_bev_features=[self.image_bev]),
            self._forward(lidar, other_bev_features=None),
        )

    def test_prepare_for_export_swaps_middle_encoder_in_place(self) -> None:
        """Test that the exportable middle encoder replaces the original and is returned."""
        lidar = self._build_lidar()
        original = lidar.pts_middle_encoder
        assert isinstance(original, _StubMiddleEncoder)

        exportable = lidar.prepare_for_export()

        assert isinstance(exportable, _StubMiddleEncoder)
        self.assertIs(exportable, lidar.pts_middle_encoder)
        self.assertIsNot(exportable, original)
        self.assertTrue(exportable.exportable)
        self.assertFalse(original.exportable)
        self.assertEqual(lidar.expected_bev_shape, self.bev_shape)


@unittest.skipUnless(
    torch.cuda.is_available(), "The sparse middle encoder runs through spconv CUDA kernels."
)
class TestBEVFusionLidarWithSparseEncoder(_BEVFusionLidarTestCase):
    """Integration tests for the lidar branch built from the real BEVFusion encoders.

    Scaled-down mirror of the TransFusion lidar stack: an 8 m range with 0.25 m voxels gives a
    32x32x40 grid, which the sparse encoder's three stride-2 stages reduce to a 4x4 BEV. The channel
    wiring follows the config (voxel encoder 32 -> sparse 128 * Z2 = 256 -> SECOND [128, 256] ->
    FPN concat 512).
    """

    def setUp(self) -> None:
        """Set up the real lidar stack and a random two-sample voxel batch on the GPU."""
        from autoware_ml.models.detection3d.encoders.sparse.sparse_encoder import SparseEncoder
        from autoware_ml.models.detection3d.encoders.voxel import HardSimpleVoxelSinCosEncoder

        super().setUp()
        self.device = torch.device("cuda:0")
        self.point_cloud_range = [0.0, 0.0, -5.0, 8.0, 8.0, 3.0]
        self.max_intensity = 255.0
        self.sparse_shape = (32, 32, 41)  # (Y, X, Z)
        self.bev_shape = (4, 4)
        self.voxel_feature_channels = 32
        self.middle_channels = 256
        self.image_channels = 80
        self.neck_channels = 512

        self.voxel_encoder = HardSimpleVoxelSinCosEncoder(
            in_channels=self.point_channels,
            min_norm_values=[*self.point_cloud_range[:3], 0.0],
            max_norm_values=[*self.point_cloud_range[3:], self.max_intensity],
        ).to(self.device)
        self.middle_encoder = SparseEncoder(
            in_channels=self.voxel_feature_channels,
            sparse_shape=list(self.sparse_shape),
            dense_output_shapes=[*self.bev_shape, 2],
        ).to(self.device)
        self.backbone = SECONDBackbone(
            in_channels=self.middle_channels,
            out_channels=[128, 256],
            layer_nums=[1, 1],
            layer_strides=[1, 2],
        ).to(self.device)
        self.neck = SECONDFPN(
            in_channels=[128, 256], out_channels=[256, 256], upsample_strides=[1, 2]
        ).to(self.device)

        num_voxels_per_sample, max_points = 64, 10
        num_voxels = self.batch_size * num_voxels_per_sample
        height, width, depth = self.sparse_shape
        # (batch, z, y, x) coordinates with distinct cells per sample.
        cells = self._distinct_cells(num_voxels_per_sample, height * width * depth)
        self.coords = torch.stack(
            [
                self._batch_indices(num_voxels_per_sample),
                cells % depth,
                cells // (depth * width),
                (cells // depth) % width,
            ],
            dim=1,
        ).int()
        point_min = torch.tensor([*self.point_cloud_range[:3], 0.0], device=self.device)
        point_max = torch.tensor(
            [*self.point_cloud_range[3:], self.max_intensity], device=self.device
        )
        self.voxels = (
            torch.rand(num_voxels, max_points, self.point_channels, device=self.device)
            * (point_max - point_min)
            + point_min
        )
        self.num_points = torch.randint(
            1, max_points + 1, (num_voxels,), dtype=torch.int32, device=self.device
        )
        self.image_bev: Float32[torch.Tensor, "batch_size channels height width"] = torch.randn(
            self.batch_size, self.image_channels, *self.bev_shape, device=self.device
        )

    def test_expected_bev_shape_matches_sparse_encoder_dense_output(self) -> None:
        """Test that the branch reports the sparse encoder's dense ``(Y, X)`` output shape."""
        self.assertEqual(self._build_lidar().expected_bev_shape, self.bev_shape)

    def test_forward_lidar_only(self) -> None:
        """Test that the real stack encodes voxels into the neck feature map."""
        bev = self._forward(self._build_lidar(), other_bev_features=None)

        self.assertEqual(bev.shape, (self.batch_size, self.neck_channels, *self.bev_shape))
        self.assertTrue(torch.isfinite(bev).all())

    def test_forward_fused_with_image_bev(self) -> None:
        """Test that an image BEV map is fused into the sparse encoder output before the backbone."""
        fuser = ConvFuser(
            in_channels=[self.image_channels, self.middle_channels],
            out_channels=self.middle_channels,
        ).to(self.device)

        bev = self._forward(self._build_lidar(fuser=fuser), other_bev_features=[self.image_bev])

        self.assertEqual(bev.shape, (self.batch_size, self.neck_channels, *self.bev_shape))
        self.assertTrue(torch.isfinite(bev).all())

    def test_prepare_for_export_keeps_lidar_output(self) -> None:
        """Test that the exportable sparse encoder reproduces the native encoder's BEV output."""
        lidar = self._build_lidar()
        reference = self._forward(lidar, other_bev_features=None)

        exportable = lidar.prepare_for_export()

        self.assertIs(exportable, lidar.pts_middle_encoder)
        self.assertIsNot(exportable, self.middle_encoder)
        self.assertEqual(lidar.expected_bev_shape, self.bev_shape)
        torch.testing.assert_close(
            self._forward(lidar, other_bev_features=None), reference, atol=1e-4, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
