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

"""Unit tests for the Depth-LSS view transform."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.view_transforms.depth_lss import (
    BEVPoolResult,
    DepthLSSNet,
    DepthLSSTransform,
    DownSampleNet,
    LidarDepthImageNet,
    _gen_dx_bx,
)

_CUDA_AVAILABLE = torch.cuda.is_available()
_REQUIRES_CUDA = "BEV pooling runs through the CUDA bev_pool kernel."


class TestGenDxBx(unittest.TestCase):
    """Unit tests for the voxel grid helper."""

    def test_returns_voxel_size_origin_and_grid_shape(self) -> None:
        """Test that voxel size, voxel center origin and grid shape follow the bounds."""
        dx, bx, nx = _gen_dx_bx(
            xbound=[-8.0, 8.0, 1.0], ybound=[-4.0, 4.0, 0.5], zbound=[-5.0, 3.0, 8.0]
        )

        self.assertEqual(dx.dtype, torch.float32)
        self.assertEqual(bx.dtype, torch.float32)
        torch.testing.assert_close(dx, torch.tensor([1.0, 0.5, 8.0]))
        torch.testing.assert_close(bx, torch.tensor([-7.5, -3.75, -1.0]))
        self.assertEqual(nx, (16, 16, 1))


class TestDownSampleNet(unittest.TestCase):
    """Unit tests for the BEV downsampling network."""

    def test_downsample_one_is_identity(self) -> None:
        """Test that a downsample factor of 1 leaves the features untouched."""
        net = DownSampleNet(downsample=1, channels=8)
        bev_features = torch.randn(2, 8, 16, 16)

        torch.testing.assert_close(net(bev_features), bev_features)

    def test_downsample_two_halves_spatial_resolution(self) -> None:
        """Test that a downsample factor of 2 halves the spatial resolution and keeps channels."""
        net = DownSampleNet(downsample=2, channels=8)

        self.assertEqual(net(torch.randn(2, 8, 16, 16)).shape, (2, 8, 8, 8))

    def test_unsupported_downsample_raises(self) -> None:
        """Test that unsupported downsample factors are rejected."""
        with self.assertRaises(ValueError):
            DownSampleNet(downsample=4, channels=8)


class TestLidarDepthImageNet(unittest.TestCase):
    """Unit tests for the lidar depth-map encoder."""

    def test_total_stride_is_four_times_last_stride(self) -> None:
        """Test that the encoder strides the depth map down by ``4 * last_stride``."""
        for last_stride, expected_size in ((1, 16), (2, 8), (4, 4)):
            with self.subTest(last_stride=last_stride):
                net = LidarDepthImageNet(in_channels=1, out_channels=64, last_stride=last_stride)

                depth_features = net(torch.randn(3, 1, 64, 64))

                self.assertEqual(depth_features.shape, (3, 64, expected_size, expected_size))


class TestDepthLSSNet(unittest.TestCase):
    """Unit tests for the depth fusion network."""

    def test_output_channels_and_resolution(self) -> None:
        """Test that the network maps to the requested channels and keeps the resolution."""
        net = DepthLSSNet(in_channels=96, out_channels=4 + 8)

        self.assertEqual(net(torch.randn(6, 96, 8, 8)).shape, (6, 12, 8, 8))


class _DepthLSSTransformTestCase(unittest.TestCase):
    """Shared helpers for the DepthLSSTransform test cases."""

    @staticmethod
    def _build_transform(
        downsample: int = 1,
        ybound: tuple[float, float, float] = (-4.0, 4.0, 1.0),
    ) -> DepthLSSTransform:
        """Build a small transform with a non-square BEV grid.

        The grid is ``nx = (X, Y, Z) = (16, 8, 1)`` so that the ``(Y, X)`` output layout can be
        told apart from the pooling-native ``(X, Y)`` layout.
        """
        return DepthLSSTransform(
            in_channels=32,
            out_channels=8,
            image_size=[64, 64],
            feature_size=[8, 8],
            xbound=[-8.0, 8.0, 1.0],
            ybound=list(ybound),
            zbound=[-5.0, 3.0, 8.0],
            dbound=[1.0, 5.0, 1.0],
            downsample=downsample,
        )

    @staticmethod
    def _identity_matrices(
        batch_size: int, num_cams: int, size: int, device: torch.device
    ) -> torch.Tensor:
        """Build a batch of identity matrices of shape ``(batch_size, num_cams, size, size)``."""
        return (
            torch.eye(size, device=device).view(1, 1, size, size).repeat(batch_size, num_cams, 1, 1)
        )


class TestDepthLSSTransform(_DepthLSSTransformTestCase):
    """Unit tests for the DepthLSSTransform that do not require the CUDA pooling kernel."""

    def setUp(self) -> None:
        """Set up a small transform."""
        torch.manual_seed(0)
        self.transform = self._build_transform()

    def test_grid_and_depth_bins_follow_bounds(self) -> None:
        """Test that the BEV grid shape and depth bin count follow the configured bounds."""
        self.assertEqual(self.transform.nx, (16, 8, 1))
        self.assertEqual(self.transform.depth_bins, 4)
        self.assertEqual(self.transform.expected_bev_shape, (8, 16))

    def test_expected_bev_shape_follows_downsample(self) -> None:
        """Test that the expected BEV shape is divided by the downsample factor."""
        self.assertEqual(self._build_transform(downsample=2).expected_bev_shape, (4, 8))

    def test_rejects_invalid_image_to_feature_stride(self) -> None:
        """Test that non-uniform strides and strides not divisible by 4 are rejected."""
        for image_size, feature_size in (([64, 32], [8, 8]), ([32, 32], [16, 16])):
            with self.subTest(image_size=image_size, feature_size=feature_size):
                with self.assertRaises(ValueError):
                    DepthLSSTransform(
                        in_channels=32,
                        out_channels=8,
                        image_size=image_size,
                        feature_size=feature_size,
                        xbound=[-8.0, 8.0, 1.0],
                        ybound=[-8.0, 8.0, 1.0],
                        zbound=[-5.0, 3.0, 8.0],
                        dbound=[1.0, 5.0, 1.0],
                    )

    def test_frustum_spans_image_plane_and_depth_bins(self) -> None:
        """Test that the frustum covers the augmented image plane at every depth bin."""
        frustum = self.transform.frustum

        self.assertEqual(frustum.shape, (4, 8, 8, 3))
        self.assertFalse(frustum.requires_grad)
        torch.testing.assert_close(frustum[..., 2].unique(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        torch.testing.assert_close(frustum[0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(frustum[-1, -1, -1], torch.tensor([63.0, 63.0, 4.0]))
        # x varies along the last spatial axis, y along the one before it.
        torch.testing.assert_close(frustum[0, 0, :, 0], torch.linspace(0.0, 63.0, 8))
        torch.testing.assert_close(frustum[0, :, 0, 1], torch.linspace(0.0, 63.0, 8))

    def test_camera_to_lidar_geometry_with_identity_matrices(self) -> None:
        """Test that identity calibration un-normalizes the frustum by its depth."""
        batch_size, num_cams = 2, 3
        device = self.transform.frustum.device

        geom = self.transform.camera_to_lidar_geometry(
            camera2aug_lidar=self._identity_matrices(batch_size, num_cams, 4, device),
            camera_intrinsics=self._identity_matrices(batch_size, num_cams, 3, device),
            img_aug_matrix=self._identity_matrices(batch_size, num_cams, 4, device),
        )

        frustum = self.transform.frustum
        expected = torch.cat([frustum[..., :2] * frustum[..., 2:3], frustum[..., 2:3]], dim=-1)
        self.assertEqual(geom.shape, (batch_size, num_cams, 4, 8, 8, 3))
        torch.testing.assert_close(geom, expected.expand(batch_size, num_cams, -1, -1, -1, -1))

    def test_camera_to_lidar_geometry_undoes_image_aug_then_applies_calibration(self) -> None:
        """
        Test one frustum point end to end: undo the image augmentation, lift it through the
        intrinsics into the camera frame, and move it into the lidar frame.
        """
        # Image augmentation: scale by 2 then translate by (1, 3) pixels.
        img_aug_matrix = torch.eye(4)
        img_aug_matrix[0, 0] = img_aug_matrix[1, 1] = 2.0
        img_aug_matrix[0, 3], img_aug_matrix[1, 3] = 1.0, 3.0
        camera_intrinsics = torch.tensor(
            [[10.0, 0.0, 32.0], [0.0, 10.0, 32.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        # Standard camera-to-lidar axis swap plus a translation.
        camera2aug_lidar = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0, 2.0],
                [0.0, -1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        geom = self.transform.camera_to_lidar_geometry(
            camera2aug_lidar=camera2aug_lidar.view(1, 1, 4, 4),
            camera_intrinsics=camera_intrinsics.view(1, 1, 3, 3),
            img_aug_matrix=img_aug_matrix.view(1, 1, 4, 4),
        )

        # Frustum point at depth bin 3, row 0, column 7 is (u, v, d) = (63, 0, 4).
        # Raw image plane: ((63 - 1) / 2, (0 - 3) / 2) = (31, -1.5).
        # Camera frame: ((31 - 32) / 10 * 4, (-1.5 - 32) / 10 * 4, 4) = (-0.4, -13.4, 4).
        # Lidar frame: (z + 1, -x + 2, -y + 3) = (5, 2.4, 16.4).
        torch.testing.assert_close(self.transform.frustum[3, 0, 7], torch.tensor([63.0, 0.0, 4.0]))
        torch.testing.assert_close(geom[0, 0, 3, 0, 7], torch.tensor([5.0, 2.4, 16.4]))

    def test_bev_pool_aux_keeps_points_inside_grid_and_sorts_by_rank(self) -> None:
        """Test that out-of-grid points are dropped and the kept points are rank sorted."""
        # Grid: x in [-8, 8), y in [-4, 4), z in [-5, 3), one meter (8 m for z) voxels.
        points = torch.tensor(
            [
                [-8.0, -4.0, 0.0],  # voxel (0, 0, 0), rank 0
                [7.9, 3.9, 0.0],  # voxel (15, 7, 0), rank 127
                [8.0, 0.0, 0.0],  # x index 16, outside
                [0.0, -5.0, 0.0],  # y index -1, outside
                [0.0, 0.0, 3.5],  # z index 1, outside
                [0.5, 0.5, -5.0],  # voxel (8, 4, 0), rank 68
            ],
            dtype=torch.float32,
        ).view(1, 1, 1, 1, 6, 3)

        result = self.transform.bev_pool_aux(points)

        self.assertIsInstance(result, BEVPoolResult)
        self.assertEqual(result.kept.tolist(), [True, True, False, False, False, True])
        self.assertEqual(result.ranks.tolist(), [0, 68, 127])
        self.assertEqual(result.indices.tolist(), [0, 2, 1])
        self.assertEqual(result.geom_feats.tolist(), [[0, 0, 0, 0], [8, 4, 0, 0], [15, 7, 0, 0]])
        self.assertEqual(result.geom_feats.dtype, torch.int64)

    def test_bev_pool_aux_assigns_batch_index_per_sample(self) -> None:
        """Test that the last geometry column carries the batch index of each frustum point."""
        batch_size, num_cams, depth_bins, height, width = 2, 2, 3, 2, 2
        # Every point sits at the grid origin, i.e. in voxel (0, 0, 0).
        points = torch.tensor([-8.0, -4.0, -5.0]).expand(
            batch_size, num_cams, depth_bins, height, width, 3
        )

        result = self.transform.bev_pool_aux(points)

        points_per_sample = num_cams * depth_bins * height * width
        self.assertTrue(result.kept.all())
        self.assertEqual(
            result.geom_feats[:, 3].tolist(),
            [0] * points_per_sample + [1] * points_per_sample,
        )
        # All points share the voxel, so only the batch index separates the ranks.
        self.assertEqual(result.ranks.unique().tolist(), [0, 1])

    def test_get_cam_feats_lifts_context_into_frustum(self) -> None:
        """Test that camera features are lifted into ``(B, N, D, fH, fW, C)`` frustum features."""
        image_features = torch.randn(2, 3, 32, 8, 8)
        depth_maps = torch.rand(2, 3, 1, 64, 64)

        feats = self.transform._get_cam_feats(image_features, depth_maps)

        self.assertEqual(feats.shape, (2, 3, 4, 8, 8, 8))


@unittest.skipUnless(_CUDA_AVAILABLE, _REQUIRES_CUDA)
class TestDepthLSSTransformBEVPooling(_DepthLSSTransformTestCase):
    """Unit tests for the DepthLSSTransform paths that pool through the CUDA kernel."""

    def setUp(self) -> None:
        """Set up a small transform and identity calibration on the GPU."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.batch_size, self.num_cams = 2, 3
        self.transform = self._build_transform().to(self.device)
        self.image_features = torch.randn(
            self.batch_size, self.num_cams, 32, 8, 8, device=self.device
        )
        self.depth_maps = torch.rand(self.batch_size, self.num_cams, 1, 64, 64, device=self.device)
        self.camera_intrinsics = self._identity_matrices(
            self.batch_size, self.num_cams, 3, self.device
        )
        self.camera2aug_lidar = self._identity_matrices(
            self.batch_size, self.num_cams, 4, self.device
        )
        self.img_aug_matrix = self._identity_matrices(
            self.batch_size, self.num_cams, 4, self.device
        )

    def _forward(self, geom_feats_precomputed: BEVPoolResult | None = None) -> torch.Tensor:
        """Run the forward pass with the shared inputs."""
        return self.transform(
            self.image_features,
            self.depth_maps,
            self.camera_intrinsics,
            self.camera2aug_lidar,
            self.img_aug_matrix,
            geom_feats_precomputed=geom_feats_precomputed,
        )

    def test_forward_emits_lidar_convention_bev_layout(self) -> None:
        """
        Test that the pooled grid comes out in the ``(Y, X)`` layout shared with the lidar
        branch rather than the pooling-native ``(X, Y)``.
        """
        bev = self._forward()

        self.assertEqual(self.transform.nx, (16, 8, 1))
        self.assertEqual(bev.shape, (self.batch_size, 8 * 1, 8, 16))
        self.assertEqual(bev.shape[-2:], self.transform.expected_bev_shape)
        self.assertTrue(torch.isfinite(bev).all())

    def test_forward_applies_downsample(self) -> None:
        """Test that the output BEV resolution is divided by the downsample factor."""
        transform = self._build_transform(downsample=2).to(self.device)

        bev = transform(
            self.image_features,
            self.depth_maps,
            self.camera_intrinsics,
            self.camera2aug_lidar,
            self.img_aug_matrix,
        )

        self.assertEqual(bev.shape, (self.batch_size, 8, 4, 8))
        self.assertEqual(bev.shape[-2:], transform.expected_bev_shape)

    def test_precomputed_metadata_matches_on_the_fly_pooling(self) -> None:
        """Test that all three pooling entry points agree on the same inputs."""
        self.transform.eval()
        geom = self.transform.camera_to_lidar_geometry(
            self.camera2aug_lidar, self.camera_intrinsics, self.img_aug_matrix
        )
        pool_result = self.transform.bev_pool_aux(geom)

        with torch.no_grad():
            bev_on_the_fly = self._forward()
            bev_precomputed_kwarg = self._forward(geom_feats_precomputed=pool_result)
            bev_precomputed = self.transform.forward_precomputed(
                self.image_features,
                self.depth_maps,
                pool_result.geom_feats,
                pool_result.kept,
                pool_result.ranks,
                pool_result.indices,
            )

        torch.testing.assert_close(bev_precomputed_kwarg, bev_on_the_fly)
        torch.testing.assert_close(bev_precomputed, bev_on_the_fly)

    def test_bev_pool_precomputed_places_features_in_their_voxel(self) -> None:
        """
        Test that a hand-built frustum lands its features in the expected ``(y, x)`` BEV cell
        and that features sharing a voxel are summed.
        """
        channels = self.transform.out_channels
        # Two points in voxel (x=2, y=5), one in voxel (x=15, y=0), one outside the grid.
        points = torch.tensor(
            [[-5.5, 1.5, 0.0], [-5.5, 1.5, 0.0], [7.5, -3.5, 0.0], [20.0, 0.0, 0.0]],
            device=self.device,
        ).view(1, 1, 1, 1, 4, 3)
        feats = torch.zeros(1, 1, 1, 1, 4, channels, device=self.device)
        feats[..., 0, :] = 1.0
        feats[..., 1, :] = 2.0
        feats[..., 2, :] = 5.0
        feats[..., 3, :] = 100.0
        pool_result = self.transform.bev_pool_aux(points)

        self.transform.eval()
        with torch.no_grad():
            bev = self.transform.bev_pool_precomputed(feats, *pool_result)

        self.assertEqual(bev.shape, (1, channels, 8, 16))
        torch.testing.assert_close(
            bev[0, :, 5, 2], torch.full((channels,), 3.0, device=self.device)
        )
        torch.testing.assert_close(
            bev[0, :, 0, 15], torch.full((channels,), 5.0, device=self.device)
        )
        self.assertAlmostEqual(bev.sum().item(), channels * (3.0 + 5.0), places=4)

    def test_training_forward_propagates_gradients(self) -> None:
        """Test that gradients flow back through pooling into the image features and depthnet."""
        self.transform.train()
        image_features = self.image_features.clone().requires_grad_(True)

        bev = self.transform(
            image_features,
            self.depth_maps,
            self.camera_intrinsics,
            self.camera2aug_lidar,
            self.img_aug_matrix,
        )
        bev.sum().backward()

        assert image_features.grad is not None
        self.assertTrue(torch.isfinite(image_features.grad).all())
        self.assertTrue(
            all(p.grad is not None for p in self.transform.depthnet.parameters() if p.requires_grad)
        )


if __name__ == "__main__":
    unittest.main()
