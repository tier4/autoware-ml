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

"""Unit tests for the BEVFusion camera branch."""

from __future__ import annotations

import unittest

from jaxtyping import Bool, Float32, Int64, UInt8
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_camera import (
    BEVFusionCamera,
    BEVFusionImageBackboneExportWrapper,
)
from autoware_ml.models.detection3d.view_transforms.depth_lss import (
    BEVPoolResult,
    DepthLSSTransform,
)


class _StubBackbone(nn.Module):
    """Single stride-8 convolution standing in for the image backbone.

    The module counts its calls so tests can check when the branch skips image encoding, and
    can return either a bare tensor or a tuple of feature maps like the real multi-scale
    backbones.
    """

    def __init__(self, feature_channels: int, return_tuple: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, feature_channels, kernel_size=8, stride=8)
        self.return_tuple = return_tuple
        self.num_calls = 0

    def forward(
        self, images: Float32[torch.Tensor, "num_images 3 height width"]
    ) -> (
        Float32[torch.Tensor, "num_images channels feature_height feature_width"]
        | tuple[Float32[torch.Tensor, "num_images channels feature_height feature_width"]]
    ):
        self.num_calls += 1
        features = self.conv(images)
        return (features,) if self.return_tuple else features


class _StubNeck(nn.Module):
    """Pass-through neck returning either a list of feature maps or the primary map only."""

    def __init__(self, return_list: bool = True) -> None:
        super().__init__()
        self.return_list = return_list

    def forward(
        self,
        features: tuple[
            Float32[torch.Tensor, "num_images channels feature_height feature_width"], ...
        ],
    ) -> (
        list[Float32[torch.Tensor, "num_images channels feature_height feature_width"]]
        | Float32[torch.Tensor, "num_images channels feature_height feature_width"]
    ):
        primary = features[0]
        return [primary, primary] if self.return_list else primary


class _PassThroughNeck(nn.Module):
    """Neck that returns the backbone output untouched."""

    def forward(
        self,
        features: tuple[
            Float32[torch.Tensor, "num_images channels feature_height feature_width"], ...
        ],
    ) -> Float32[torch.Tensor, "num_images channels feature_height feature_width"]:
        return features[0]


class _BEVFusionCameraTestCase(unittest.TestCase):
    """Shared configuration and helpers for the camera branch test cases."""

    def setUp(self) -> None:
        """Set up the device and the image and BEV geometry shared by every camera branch test."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = 64
        self.feature_size = 8
        self.feature_channels = 32
        self.bev_channels = 8
        # The view transform grid is nx = (X, Y, Z) = (16, 8, 1), so the (Y, X) output layout can
        # be told apart from the pooling-native (X, Y) layout.
        self.bev_shape = (8, 16)
        self.xbound = [-8.0, 8.0, 1.0]
        self.ybound = [-4.0, 4.0, 1.0]
        self.zbound = [-5.0, 3.0, 8.0]
        self.dbound = [1.0, 5.0, 1.0]

    def _build_view_transform(self) -> DepthLSSTransform:
        """Build a small view transform with the non-square grid from ``setUp``."""
        return DepthLSSTransform(
            in_channels=self.feature_channels,
            out_channels=self.bev_channels,
            image_size=[self.image_size, self.image_size],
            feature_size=[self.feature_size, self.feature_size],
            xbound=self.xbound,
            ybound=self.ybound,
            zbound=self.zbound,
            dbound=self.dbound,
        ).to(self.device)

    def _build_camera(
        self, img_backbone: nn.Module | None = None, img_neck: nn.Module | None = None
    ) -> BEVFusionCamera:
        """Build a camera branch from stub image modules and the small view transform."""
        return BEVFusionCamera(
            img_backbone=(
                _StubBackbone(self.feature_channels).to(self.device)
                if img_backbone is None
                else img_backbone
            ),
            img_neck=_StubNeck().to(self.device) if img_neck is None else img_neck,
            view_transform=self._build_view_transform(),
        ).to(self.device)

    def _identity_matrices(
        self, batch_size: int, num_cams: int, size: int
    ) -> Float32[torch.Tensor, "batch_size num_cams size size"]:
        """Build a batch of identity matrices of shape ``(batch_size, num_cams, size, size)``."""
        return (
            torch.eye(size, device=self.device)
            .view(1, 1, size, size)
            .repeat(batch_size, num_cams, 1, 1)
        )

    def _calibration(
        self, batch_size: int, num_cams: int
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_cams 3 3"],
        Float32[torch.Tensor, "batch_size num_cams 4 4"],
    ]:
        """Build intrinsics and augmented lidar-to-camera extrinsics that look into the BEV grid.

        The pinhole has its principal point at the image center with a focal length of half the
        image size, so the frustum spans ``[-d, d)`` metres at depth ``d``. The extrinsics apply
        the standard camera-to-lidar axis swap (camera z becomes lidar x) and shift each camera
        along lidar y, so the projected frustum lands inside the ``x in [-8, 8), y in [-4, 4)``
        grid and differs per camera.
        """
        focal = self.image_size / 2.0
        camera_intrinsics = torch.tensor(
            [[focal, 0.0, focal], [0.0, focal, focal], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        camera_intrinsics = camera_intrinsics.view(1, 1, 3, 3).repeat(batch_size, num_cams, 1, 1)

        camera2lidar = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        camera2lidar = camera2lidar.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1).clone()
        camera2lidar[:, :, 1, 3] = torch.linspace(-1.0, 1.0, num_cams, device=self.device)
        aug_lidar2cam = torch.inverse(camera2lidar)
        return camera_intrinsics, aug_lidar2cam


class TestBEVFusionCamera(_BEVFusionCameraTestCase):
    """Unit tests for the camera branch paths that do not require the CUDA pooling kernel."""

    def setUp(self) -> None:
        """Set up a camera branch built from stub image modules."""
        super().setUp()
        self.batch_size, self.num_cams = 2, 3
        self.camera = self._build_camera()
        self.image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"] = torch.randn(
            self.batch_size, self.num_cams, 3, self.image_size, self.image_size, device=self.device
        )

    def test_expected_bev_shape_delegates_to_view_transform(self) -> None:
        """Test that the branch reports the ``(height, width)`` BEV grid of its view transform."""
        self.assertEqual(self.camera.expected_bev_shape, self.bev_shape)
        self.assertEqual(
            self.camera.expected_bev_shape, self.camera.view_transform.expected_bev_shape
        )

    def test_extract_image_features_restores_camera_dimension(self) -> None:
        """Test that images are encoded per camera and folded back into ``(B, N, C, fH, fW)``."""
        image_features = self.camera.extract_image_features(self.image_batch)

        self.assertEqual(
            image_features.shape,
            (
                self.batch_size,
                self.num_cams,
                self.feature_channels,
                self.feature_size,
                self.feature_size,
            ),
        )
        # Every (batch, camera) slot holds the features of exactly that image.
        expected = self.camera.img_backbone(self.image_batch[1, 2:3])[0]
        torch.testing.assert_close(image_features[1, 2], expected)

    def test_extract_image_features_accepts_tensor_or_sequence_outputs(self) -> None:
        """Test that tensor and tuple backbone outputs, and tensor and list neck outputs, agree."""
        reference = self.camera.extract_image_features(self.image_batch)

        for return_tuple in (False, True):
            for return_list in (False, True):
                with self.subTest(backbone_tuple=return_tuple, neck_list=return_list):
                    backbone = _StubBackbone(self.feature_channels, return_tuple=return_tuple).to(
                        self.device
                    )
                    backbone.load_state_dict(self.camera.img_backbone.state_dict())
                    camera = self._build_camera(
                        img_backbone=backbone,
                        img_neck=_StubNeck(return_list=return_list).to(self.device),
                    )

                    torch.testing.assert_close(
                        camera.extract_image_features(self.image_batch), reference
                    )

    def test_build_export_geometry_matches_view_transform_with_identity_image_aug(self) -> None:
        """
        Test that the export geometry equals the view transform's pooling metadata computed from
        the inverted extrinsics and an identity image augmentation.
        """
        camera_intrinsics, aug_lidar2cam = self._calibration(1, self.num_cams)
        view_transform = self.camera.view_transform

        geom_feats, kept, ranks, indices = self.camera.build_export_geometry(
            camera_intrinsics, aug_lidar2cam
        )

        expected = view_transform.bev_pool_aux(
            view_transform.camera_to_lidar_geometry(
                torch.inverse(aug_lidar2cam),
                camera_intrinsics,
                self._identity_matrices(1, self.num_cams, 4),
            )
        )
        self.assertIsInstance(expected, BEVPoolResult)
        torch.testing.assert_close(geom_feats, expected.geom_feats)
        torch.testing.assert_close(kept, expected.kept)
        torch.testing.assert_close(ranks, expected.ranks)
        torch.testing.assert_close(indices, expected.indices)

    def test_build_export_geometry_layout(self) -> None:
        """Test dtypes, sizes and rank ordering of the exported pooling metadata."""
        camera_intrinsics, aug_lidar2cam = self._calibration(1, self.num_cams)
        view_transform = self.camera.view_transform
        num_frustum_points = self.num_cams * view_transform.depth_bins * self.feature_size**2

        geom_feats, kept, ranks, indices = self.camera.build_export_geometry(
            camera_intrinsics, aug_lidar2cam
        )

        self.assertEqual(kept.dtype, torch.bool)
        self.assertEqual(kept.shape, (num_frustum_points,))
        num_kept = int(kept.sum())
        self.assertGreater(num_kept, 0)
        self.assertLess(num_kept, num_frustum_points)
        self.assertEqual(geom_feats.dtype, torch.int64)
        self.assertEqual(geom_feats.shape, (num_kept, 4))
        self.assertEqual(ranks.shape, (num_kept,))
        self.assertEqual(indices.shape, (num_kept,))
        self.assertTrue(torch.all(ranks[1:] >= ranks[:-1]))
        # A single exported sample only ever carries batch index 0.
        self.assertTrue(torch.all(geom_feats[:, 3] == 0))
        # Kept points lie inside the (X, Y, Z) = (16, 8, 1) grid.
        self.assertTrue(torch.all(geom_feats[:, :3] >= 0))
        self.assertTrue(
            torch.all(geom_feats[:, :3] < torch.tensor(view_transform.nx, device=self.device))
        )


class TestBEVFusionImageBackboneExportWrapper(_BEVFusionCameraTestCase):
    """Unit tests for the image backbone export wrapper."""

    def setUp(self) -> None:
        """Set up raw ``uint8`` multiview images for a single sample."""
        super().setUp()
        self.num_cams = 3
        self.imgs: UInt8[torch.Tensor, "num_cams 3 height width"] = torch.randint(
            0,
            256,
            (self.num_cams, 3, self.image_size, self.image_size),
            dtype=torch.uint8,
            device=self.device,
        )

    def test_normalizes_uint8_images_by_255(self) -> None:
        """Test that the wrapper bakes the ``1 / 255`` normalization into the graph."""
        camera = self._build_camera(
            img_backbone=nn.Identity().to(self.device), img_neck=_PassThroughNeck().to(self.device)
        )
        wrapper = BEVFusionImageBackboneExportWrapper(camera).to(self.device)
        imgs = self.imgs.clone()
        imgs[0] = 0
        imgs[1] = 255

        normalized = wrapper(imgs)

        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(normalized.shape, imgs.shape)
        torch.testing.assert_close(normalized, imgs.float() / 255.0)
        self.assertEqual(normalized[0].min().item(), 0.0)
        self.assertEqual(normalized[1].max().item(), 1.0)

    def test_output_matches_extract_image_features_without_batch_dim(self) -> None:
        """Test that the wrapper returns the single-sample neck features as ``(N, C, fH, fW)``."""
        camera = self._build_camera().eval()
        wrapper = BEVFusionImageBackboneExportWrapper(camera).to(self.device)

        with torch.no_grad():
            features = wrapper(self.imgs)
            expected = camera.extract_image_features((self.imgs.float() / 255.0).unsqueeze(0))

        self.assertEqual(
            features.shape,
            (self.num_cams, self.feature_channels, self.feature_size, self.feature_size),
        )
        torch.testing.assert_close(features, expected[0])


@unittest.skipUnless(
    torch.cuda.is_available(), "BEV pooling runs through the CUDA bev_pool kernel."
)
class TestBEVFusionCameraForward(_BEVFusionCameraTestCase):
    """Unit tests for the camera branch paths that pool through the CUDA kernel."""

    def setUp(self) -> None:
        """Set up a camera branch and calibrated multiview inputs on the GPU."""
        super().setUp()
        self.batch_size, self.num_cams = 2, 3
        self.camera = self._build_camera().eval()
        self.image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"] = torch.randn(
            self.batch_size,
            self.num_cams,
            3,
            self.image_size,
            self.image_size,
            device=self.device,
        )
        self.depth_maps: Float32[torch.Tensor, "batch_size num_cams 1 height width"] = torch.rand(
            self.batch_size, self.num_cams, 1, self.image_size, self.image_size, device=self.device
        )
        self.camera_intrinsics, self.aug_lidar2cam = self._calibration(
            self.batch_size, self.num_cams
        )

    def _forward(
        self,
        image_features: Float32[
            torch.Tensor, "batch_size num_cams channels feature_height feature_width"
        ]
        | None = None,
        img_aug_matrix: Float32[torch.Tensor, "batch_size num_cams 4 4"] | None = None,
        geom_feats_precomputed: BEVPoolResult | None = None,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Run the forward pass with the shared inputs and the given optional overrides."""
        with torch.no_grad():
            return self.camera(
                image_batch=self.image_batch,
                depth_maps=self.depth_maps,
                camera_intrinsics=self.camera_intrinsics,
                aug_lidar2cam=self.aug_lidar2cam,
                geom_feats_precomputed=geom_feats_precomputed,
                image_features=image_features,
                img_aug_matrix=img_aug_matrix,
            )

    def test_forward_emits_bev_of_expected_shape(self) -> None:
        """Test that the branch pools the images into a ``(B, C, Y, X)`` BEV map."""
        bev = self._forward()

        self.assertEqual(bev.shape, (self.batch_size, self.bev_channels, *self.bev_shape))
        self.assertEqual(bev.shape[-2:], self.camera.expected_bev_shape)
        self.assertTrue(torch.isfinite(bev).all())
        self.assertGreater(bev.abs().sum().item(), 0.0)

    def test_forward_skips_backbone_when_image_features_are_given(self) -> None:
        """Test that precomputed image features bypass the backbone and give the same BEV map."""
        reference = self._forward()
        with torch.no_grad():
            image_features = self.camera.extract_image_features(self.image_batch)
        backbone = self.camera.img_backbone
        assert isinstance(backbone, _StubBackbone)
        backbone.num_calls = 0

        bev = self._forward(image_features=image_features)

        self.assertEqual(backbone.num_calls, 0)
        torch.testing.assert_close(bev, reference)

    def test_forward_defaults_to_identity_image_augmentation(self) -> None:
        """Test that omitting ``img_aug_matrix`` equals passing identity augmentation matrices."""
        identity = self._identity_matrices(self.batch_size, self.num_cams, 4)

        torch.testing.assert_close(self._forward(), self._forward(img_aug_matrix=identity))

    def test_forward_applies_image_augmentation(self) -> None:
        """Test that a non-identity image augmentation changes where features are pooled."""
        img_aug_matrix = self._identity_matrices(self.batch_size, self.num_cams, 4)
        img_aug_matrix[..., 0, 3] = 16.0  # Shift the image by 16 pixels along x.

        self.assertFalse(
            torch.allclose(self._forward(), self._forward(img_aug_matrix=img_aug_matrix))
        )

    def test_forward_uses_precomputed_geometry(self) -> None:
        """Test that precomputed pooling metadata reproduces the on-the-fly result."""
        view_transform = self.camera.view_transform
        pool_result = view_transform.bev_pool_aux(
            view_transform.camera_to_lidar_geometry(
                torch.inverse(self.aug_lidar2cam),
                self.camera_intrinsics,
                self._identity_matrices(self.batch_size, self.num_cams, 4),
            )
        )

        torch.testing.assert_close(
            self._forward(geom_feats_precomputed=pool_result), self._forward()
        )

    def test_forward_export_matches_forward_for_single_sample(self) -> None:
        """
        Test that the single-sample export path, fed batch-less inputs and float geometry, equals
        the batched forward pass of the first sample.
        """
        with torch.no_grad():
            reference = self.camera(
                image_batch=self.image_batch[:1],
                depth_maps=self.depth_maps[:1],
                camera_intrinsics=self.camera_intrinsics[:1],
                aug_lidar2cam=self.aug_lidar2cam[:1],
            )
            image_features = self.camera.extract_image_features(self.image_batch[:1])[0]
            geom_feats: Int64[torch.Tensor, "num_kept 4"]
            kept: Bool[torch.Tensor, " num_frustum_points"]
            ranks: Int64[torch.Tensor, " num_kept"]
            indices: Int64[torch.Tensor, " num_kept"]
            geom_feats, kept, ranks, indices = self.camera.build_export_geometry(
                self.camera_intrinsics[:1], self.aug_lidar2cam[:1]
            )

            bev = self.camera.forward_export(
                image_features=image_features,
                depth_maps=self.depth_maps[0],
                geom_feats=geom_feats.float(),
                kept=kept,
                ranks=ranks,
                indices=indices,
            )

        self.assertEqual(bev.shape, (1, self.bev_channels, *self.bev_shape))
        torch.testing.assert_close(bev, reference)


if __name__ == "__main__":
    unittest.main()
