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

"""Unit tests for the LidarCameraFusion, Affine and SaveFusionPreview transforms."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.camera_lidar.camera_lidar import (
    Affine,
    LidarCameraFusion,
    SaveFusionPreview,
)
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.utils.calibration import CalibrationStatus

HEIGHT, WIDTH = 8, 12
FOCAL, CENTER_X, CENTER_Y = 10.0, 6.0, 4.0
RGB_VALUE = 100.0 / 255.0  # Images arrive in unit range from the loader


def _intrinsics(num_cameras: int) -> torch.Tensor:
    return torch.tensor(
        [[FOCAL, 0.0, CENTER_X], [0.0, FOCAL, CENTER_Y], [0.0, 0.0, 1.0]], dtype=torch.float32
    ).repeat(num_cameras, 1, 1)


def _homogeneous(intrinsics: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.eye(4, dtype=intrinsics.dtype).repeat(intrinsics.shape[0], 1, 1)
    homogeneous[:, :3, :3] = intrinsics
    return homogeneous


def _build_sample(
    points_xyzi: torch.Tensor,
    num_cameras: int = 1,
    lidar2cams: torch.Tensor | None = None,
    noises: torch.Tensor | None = None,
    calibration_statuses: torch.Tensor | None = None,
    distortion_coefficients: torch.Tensor | None = None,
    images: torch.Tensor | None = None,
) -> ModelGTSample:
    """Build a sample whose cameras look along +z of the lidar frame (identity extrinsics)."""
    intrinsics = _intrinsics(num_cameras)
    if lidar2cams is None:
        lidar2cams = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
    if images is None:
        images = torch.full((num_cameras, 3, HEIGHT, WIDTH), RGB_VALUE, dtype=torch.float32)
    coefficients = torch.zeros(0) if distortion_coefficients is None else distortion_coefficients
    camera_image_data = BaseImages(
        images=images,
        timestamps=torch.zeros(num_cameras, dtype=torch.float64),
        camera_intrinsics=intrinsics,
        camera_names=[f"camera{index}" for index in range(num_cameras)],
        lidar2images=_homogeneous(intrinsics) @ lidar2cams,
        lidar2cams=lidar2cams,
        distortion_models=[""] * num_cameras,
        distortion_coefficients=[coefficients.clone() for _ in range(num_cameras)],
        augmented_camera_intrinsics=intrinsics.clone(),
        image_augmentation_matrices=torch.eye(4).repeat(num_cameras, 1, 1),
        noises=noises,
        calibration_statuses=calibration_statuses,
    )
    image_samples = [
        ImageSample(
            image_path=f"/data/frame_{index}.jpg",
            camera_name=f"camera{index}",
            timestamp=0.0,
            camera_intrinsic=intrinsics[index],
            lidar2cam=lidar2cams[index],
            lidar2image=camera_image_data.lidar2images[index],
            distortion_model="",
            distortion_coefficients=coefficients.clone(),
        )
        for index in range(num_cameras)
    ]
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=image_samples,
        point_cloud_data=LiDARPoints(
            points=points_xyzi,
            point_feature_names=[
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        ),
        camera_image_data=camera_image_data,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )


def _pixel_of(x: float, z: float) -> int:
    """Column hit by a point at lateral offset ``x`` and depth ``z`` (camera on the lidar)."""
    return int(CENTER_X + FOCAL * x / z)


class TestLidarCameraFusion(unittest.TestCase):
    """Fusion of the lidar points into the camera images."""

    def setUp(self) -> None:
        self.fusion = LidarCameraFusion(max_depth=128.0, dilation_size=0)

    def test_output_layout_and_rgb_passthrough(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 0.2]]), num_cameras=2)

        output = self.fusion(sample).camera_image_data

        assert output.depth_maps is not None
        self.assertEqual(tuple(output.depth_maps.shape), (2, 2, HEIGHT, WIDTH))
        self.assertEqual(output.depth_maps.dtype, torch.float32)
        # Only the depth maps change.
        torch.testing.assert_close(output.images, sample.camera_image_data.images)
        torch.testing.assert_close(output.lidar2images, sample.camera_image_data.lidar2images)

    def test_points_land_on_expected_pixels_with_scaled_depth_and_intensity(self) -> None:
        points = torch.tensor([[0.0, 0.0, 5.0, 0.2], [1.0, 0.0, 5.0, 1.0]])
        sample = _build_sample(points)

        output = self.fusion(sample).camera_image_data

        depth = output.depth_maps[0, 0]
        intensity = output.depth_maps[0, 1]
        center = (int(CENTER_Y), int(CENTER_X))
        right = (int(CENTER_Y), _pixel_of(1.0, 5.0))
        self.assertAlmostEqual(float(depth[center]), 5.0 / 128.0, places=5)
        self.assertAlmostEqual(float(depth[right]), 5.0 / 128.0, places=5)
        self.assertAlmostEqual(float(intensity[center]), 0.2, places=5)
        self.assertAlmostEqual(float(intensity[right]), 1.0, places=5)
        self.assertEqual(int((depth > 0).sum()), 2)
        self.assertEqual(int((intensity > 0).sum()), 2)

    def test_depth_follows_max_depth_and_intensity_is_clipped_to_unit_range(self) -> None:
        fusion = LidarCameraFusion(max_depth=10.0, dilation_size=0)
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 127.5]]))

        output = fusion(sample).camera_image_data

        center = (int(CENTER_Y), int(CENTER_X))
        self.assertAlmostEqual(float(output.depth_maps[0, 0][center]), 0.5, places=5)
        self.assertAlmostEqual(float(output.depth_maps[0, 1][center]), 1.0, places=5)

    def test_points_behind_camera_beyond_max_depth_or_off_image_are_dropped(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0, -5.0, 1.0],  # behind the camera
                [0.0, 0.0, 200.0, 1.0],  # beyond max_depth
                [50.0, 0.0, 5.0, 1.0],  # projects outside the image
            ]
        )
        sample = _build_sample(points)

        output = self.fusion(sample).camera_image_data

        self.assertEqual(float(output.depth_maps.abs().sum()), 0.0)

    def test_empty_point_cloud_gives_zero_channels(self) -> None:
        sample = _build_sample(torch.zeros((0, 4)))

        output = self.fusion(sample).camera_image_data

        self.assertEqual(tuple(output.depth_maps.shape), (1, 2, HEIGHT, WIDTH))
        self.assertEqual(float(output.depth_maps.abs().sum()), 0.0)

    def test_dilation_paints_square_patch(self) -> None:
        fusion = LidarCameraFusion(dilation_size=1)
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))

        output = fusion(sample).camera_image_data

        hit = output.depth_maps[0, 0] > 0
        self.assertEqual(int(hit.sum()), 9)
        rows, cols = torch.nonzero(hit, as_tuple=True)
        self.assertEqual((rows.min().item(), rows.max().item()), (CENTER_Y - 1, CENTER_Y + 1))
        self.assertEqual((cols.min().item(), cols.max().item()), (CENTER_X - 1, CENTER_X + 1))

    def test_closest_point_wins_on_shared_pixel(self) -> None:
        points = torch.tensor([[0.0, 0.0, 20.0, 0.1], [0.0, 0.0, 5.0, 0.8]])
        sample = _build_sample(points)

        output = self.fusion(sample).camera_image_data

        center = (int(CENTER_Y), int(CENTER_X))
        self.assertAlmostEqual(float(output.depth_maps[0, 0][center]), 5.0 / 128.0, places=5)
        self.assertAlmostEqual(float(output.depth_maps[0, 1][center]), 0.8, places=5)

    def test_projection_uses_augmented_intrinsics(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))
        # Emulate an image-space shift of two pixels to the left, as CropAndScale would leave it.
        shifted = sample.camera_image_data.augmented_camera_intrinsics.clone()
        shifted[:, 0, 2] -= 2.0
        camera_image_data = BaseImages.model_validate(
            sample.camera_image_data.model_copy(
                update={
                    "augmented_camera_intrinsics": shifted,
                    "lidar2images": _homogeneous(shifted) @ sample.camera_image_data.lidar2cams,
                }
            )
        )
        sample = sample._replace(camera_image_data=camera_image_data)

        output = self.fusion(sample).camera_image_data

        hit = torch.nonzero(output.depth_maps[0, 0] > 0)
        self.assertEqual(hit.tolist(), [[int(CENTER_Y), int(CENTER_X) - 2]])

    def test_zero_distortion_coefficients_match_pinhole(self) -> None:
        points = torch.tensor([[1.0, 0.5, 5.0, 1.0], [-1.0, -0.5, 8.0, 0.4]])
        pinhole = self.fusion(_build_sample(points)).camera_image_data.depth_maps
        with_zeros = self.fusion(
            _build_sample(points, distortion_coefficients=torch.zeros(5))
        ).camera_image_data.depth_maps

        torch.testing.assert_close(pinhole, with_zeros)

    def test_ego_box_drops_occluded_points(self) -> None:
        fusion = LidarCameraFusion(dilation_size=0, ego_box=[-0.2, -0.2, 1.0, 0.2, 0.2, 2.0])
        # The first ray passes through the box, the second passes beside it.
        points = torch.tensor([[0.0, 0.0, 5.0, 1.0], [2.0, 0.0, 5.0, 1.0]])
        sample = _build_sample(points)

        output = fusion(sample).camera_image_data

        hit = torch.nonzero(output.depth_maps[0, 0] > 0)
        self.assertEqual(hit.tolist(), [[int(CENTER_Y), _pixel_of(2.0, 5.0)]])

    def test_camera_inside_ego_box_keeps_forward_points(self) -> None:
        fusion = LidarCameraFusion(dilation_size=0, ego_box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))

        output = fusion(sample).camera_image_data

        self.assertEqual(int((output.depth_maps[0, 0] > 0).sum()), 1)

    def test_camera_center_undoes_misalignment_noise(self) -> None:
        true_lidar2cam = np.eye(4)
        true_lidar2cam[:3, 3] = [0.0, 0.0, -2.0]  # camera sits at z=2 in the lidar frame
        noise = np.eye(4)
        noise[:3, 3] = [1.0, 0.0, 0.0]
        noisy_lidar2cam = noise @ true_lidar2cam

        center = LidarCameraFusion.camera_center_in_lidar(noisy_lidar2cam, noise)
        naive = LidarCameraFusion.camera_center_in_lidar(noisy_lidar2cam, None)

        np.testing.assert_allclose(center, [0.0, 0.0, 2.0])
        np.testing.assert_allclose(naive, [-1.0, 0.0, 2.0])

    def test_occlusion_uses_true_camera_center_when_noisy(self) -> None:
        # A pure x-translation noise moves the projection but must not move the ego box test.
        fusion = LidarCameraFusion(dilation_size=0, ego_box=[-0.2, -0.2, 1.0, 0.2, 0.2, 2.0])
        noise = torch.eye(4).unsqueeze(0)
        noise[0, 0, 3] = -0.6
        lidar2cams = noise @ torch.eye(4).unsqueeze(0)
        sample = _build_sample(
            torch.tensor([[0.0, 0.0, 5.0, 1.0]]), lidar2cams=lidar2cams, noises=noise
        )

        output = fusion(sample).camera_image_data

        # Occluded from the true camera position, so nothing is painted.
        self.assertEqual(float(output.depth_maps[0, 0].abs().sum()), 0.0)

    def test_missing_keys(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))

        with self.assertRaisesRegex(KeyError, "Missing required key 'point_cloud_data'"):
            self.fusion(sample._replace(point_cloud_data=None))
        with self.assertRaisesRegex(KeyError, "Missing required key 'camera_image_data'"):
            self.fusion(sample._replace(camera_image_data=None))

    def test_missing_intensity_feature(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))
        xyz_only = LiDARPoints(
            points=sample.point_cloud_data.points[:, :3],
            point_feature_names=[PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z],
            timestamp=0.0,
        )

        with self.assertRaisesRegex(ValueError, "intensity feature"):
            self.fusion(sample._replace(point_cloud_data=xyz_only))

    def test_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "ego_box must hold six values"):
            LidarCameraFusion(ego_box=[0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "max_depth must be positive"):
            LidarCameraFusion(max_depth=0.0)
        with self.assertRaisesRegex(ValueError, "dilation_size must be >= 0"):
            LidarCameraFusion(dilation_size=-1)

    def test_input_sample_is_not_mutated(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))
        original_images = sample.camera_image_data.images.clone()

        self.fusion(sample)

        torch.testing.assert_close(sample.camera_image_data.images, original_images)
        self.assertIsNone(sample.camera_image_data.depth_maps)


class TestAffine(unittest.TestCase):
    """Random affine warp of the camera images."""

    def setUp(self) -> None:
        np.random.seed(0)
        gradient = torch.linspace(0.0, 1.0, WIDTH).repeat(HEIGHT, 1)
        images = torch.stack([gradient, gradient.flip(1), gradient * 0.5]).unsqueeze(0)
        self.sample = _build_sample(
            torch.tensor([[0.0, 0.0, 5.0, 1.0]]), num_cameras=2, images=images.repeat(2, 1, 1, 1)
        )

    def test_probability_zero_leaves_sample_unchanged(self) -> None:
        output = Affine(probability=0.0)(self.sample)

        self.assertIs(output, self.sample)

    def test_warp_composes_affine_into_calibration(self) -> None:
        original = self.sample.camera_image_data

        output = Affine(probability=1.0, max_distortion=0.1)(self.sample).camera_image_data

        self.assertEqual(tuple(output.images.shape), tuple(original.images.shape))
        self.assertFalse(torch.allclose(output.images, original.images))
        # Recover the pixel affine from the intrinsics and check every field agrees with it.
        affines = output.augmented_camera_intrinsics @ torch.linalg.inv(
            original.augmented_camera_intrinsics
        )
        torch.testing.assert_close(affines[:, 2], torch.tensor([0.0, 0.0, 1.0]).repeat(2, 1))
        torch.testing.assert_close(output.image_augmentation_pixel_affines(), affines)
        torch.testing.assert_close(
            output.lidar2images,
            _homogeneous(output.augmented_camera_intrinsics) @ output.lidar2cams,
        )
        torch.testing.assert_close(output.lidar2cams, original.lidar2cams)
        torch.testing.assert_close(output.camera_intrinsics, original.camera_intrinsics)

    def test_each_camera_draws_its_own_affine(self) -> None:
        output = Affine(probability=1.0, max_distortion=0.1)(self.sample).camera_image_data

        self.assertFalse(
            torch.allclose(
                output.augmented_camera_intrinsics[0], output.augmented_camera_intrinsics[1]
            )
        )

    def test_zoom_keeps_viewport_inside_source_image(self) -> None:
        transform = Affine(probability=1.0, max_distortion=0.2)
        corners = np.array(
            [[0, 0, 1], [WIDTH, 0, 1], [WIDTH, HEIGHT, 1], [0, HEIGHT, 1]], dtype=np.float64
        )

        for _ in range(50):
            affine = transform.sample_affine_matrix(HEIGHT, WIDTH)

            np.testing.assert_allclose(affine[2], [0.0, 0.0, 1.0])
            source_corners = corners @ np.linalg.inv(affine).T
            self.assertTrue(np.all(source_corners[:, 0] >= -1e-6))
            self.assertTrue(np.all(source_corners[:, 0] <= WIDTH + 1e-6))
            self.assertTrue(np.all(source_corners[:, 1] >= -1e-6))
            self.assertTrue(np.all(source_corners[:, 1] <= HEIGHT + 1e-6))

    def test_constant_image_has_no_border_artifacts(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]), num_cameras=1)

        output = Affine(probability=1.0, max_distortion=0.2)(sample).camera_image_data

        torch.testing.assert_close(
            output.images, torch.full_like(output.images, RGB_VALUE), atol=1e-5, rtol=0.0
        )

    def test_fusion_after_affine_follows_the_warped_pixels(self) -> None:
        # The projection of a lidar point must move with the pixels it was drawn on.
        point = torch.tensor([[1.0, 0.5, 5.0, 1.0]])
        sample = _build_sample(point)
        fusion = LidarCameraFusion(dilation_size=0)

        warped = Affine(probability=1.0, max_distortion=0.1)(sample)
        fused = fusion(warped).camera_image_data

        affine = warped.camera_image_data.image_augmentation_pixel_affines()[0]
        raw_pixel = torch.tensor([_pixel_of(1.0, 5.0) + 0.0, CENTER_Y + FOCAL * 0.5 / 5.0, 1.0])
        expected = (affine @ raw_pixel)[:2]
        hit = torch.nonzero(fused.depth_maps[0, 0] > 0)
        self.assertEqual(hit.tolist(), [[int(expected[1]), int(expected[0])]])

    def test_missing_camera_image_data(self) -> None:
        with self.assertRaisesRegex(KeyError, "Missing required key 'camera_image_data'"):
            Affine()(self.sample._replace(camera_image_data=None))


class TestSaveFusionPreview(unittest.TestCase):
    """Preview images of the projected points."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp_dir.name) / "previews"
        points = torch.tensor([[0.0, 0.0, 5.0, 0.2], [1.0, 0.0, 5.0, 1.0]])
        self.fused_sample = LidarCameraFusion(dilation_size=0)(_build_sample(points, num_cameras=2))

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _with_statuses(self, statuses: torch.Tensor | None) -> ModelGTSample:
        camera_image_data = BaseImages.model_validate(
            self.fused_sample.camera_image_data.model_copy(
                update={"calibration_statuses": statuses}
            )
        )
        return self.fused_sample._replace(camera_image_data=camera_image_data)

    def test_writes_two_previews_per_camera_named_after_camera_and_timestamp(self) -> None:
        transform = SaveFusionPreview(probability=1.0, out_dir=self.out_dir)

        output = transform(self.fused_sample)

        self.assertIs(output, self.fused_sample)
        self.assertEqual(
            sorted(path.name for path in self.out_dir.iterdir()),
            [
                "camera0_0.000000_depth.png",
                "camera0_0.000000_intensity.png",
                "camera1_0.000000_depth.png",
                "camera1_0.000000_intensity.png",
            ],
        )

    def test_status_suffix(self) -> None:
        statuses = torch.tensor(
            [CalibrationStatus.CALIBRATED.value, CalibrationStatus.MISCALIBRATED.value]
        )
        SaveFusionPreview(probability=1.0, out_dir=self.out_dir)(self._with_statuses(statuses))

        names = sorted(path.name for path in self.out_dir.iterdir())
        self.assertIn("camera0_0.000000_calibrated_depth.png", names)
        self.assertIn("camera1_0.000000_miscalibrated_intensity.png", names)

    def test_probability_zero_writes_nothing(self) -> None:
        SaveFusionPreview(probability=0.0, out_dir=self.out_dir)(self.fused_sample)

        self.assertEqual(list(self.out_dir.iterdir()), [])

    def test_recover_channels_inverts_fusion_scaling(self) -> None:
        transform = SaveFusionPreview(out_dir=self.out_dir, max_depth=128.0)
        camera_image_data = self.fused_sample.camera_image_data

        rgb, depth, intensity = transform.recover_channels(
            camera_image_data.images[0].numpy(), camera_image_data.depth_maps[0].numpy()
        )

        self.assertEqual(rgb.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb == 100))
        self.assertAlmostEqual(float(depth[int(CENTER_Y), int(CENTER_X)]), 5.0, places=4)
        self.assertAlmostEqual(float(intensity[int(CENTER_Y), int(CENTER_X)]), 0.2, places=4)

    def test_overlay_only_touches_pixels_with_points(self) -> None:
        transform = SaveFusionPreview(out_dir=self.out_dir, alpha=1.0)
        rgb = np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8)
        values = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        values[2, 3] = 1.0

        overlay = transform.create_overlay(rgb, values, transform.depth_cmap, alpha=1.0)

        untouched = np.ones((HEIGHT, WIDTH), dtype=bool)
        untouched[2, 3] = False
        self.assertTrue(np.all(overlay[untouched] == 7))
        self.assertFalse(np.all(overlay[2, 3] == 7))

    def test_rejects_images_without_depth_maps(self) -> None:
        sample = _build_sample(torch.tensor([[0.0, 0.0, 5.0, 1.0]]))

        with self.assertRaisesRegex(ValueError, "no depth maps"):
            SaveFusionPreview(out_dir=self.out_dir)(sample)

    def test_missing_camera_image_data(self) -> None:
        with self.assertRaisesRegex(KeyError, "Missing required key 'camera_image_data'"):
            SaveFusionPreview(out_dir=self.out_dir)(
                self.fused_sample._replace(camera_image_data=None)
            )


if __name__ == "__main__":
    unittest.main()
