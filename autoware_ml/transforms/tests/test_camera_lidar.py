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

"""Unit tests for the camera lidar transforms running on ``ModelGTSample``."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.camera.resize import CropAndScale
from autoware_ml.transforms.camera_lidar.camera_lidar import (
    Affine,
    AxisActivation,
    CalibrationMisalignment,
    LidarCameraFusion,
    MagnitudeRange,
    SaveFusionPreview,
)
from autoware_ml.transforms.point_cloud.geometry import CropBoxInner
from autoware_ml.transforms.tests.test_camera import build_images, build_sample
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.utils.calibration import CalibrationStatus

MISALIGNMENT_RANGES = {
    "activate": AxisActivation(yaw=True),
    "yaw": MagnitudeRange(min_neg=4.0, max_neg=8.0, min_pos=4.0, max_pos=8.0),
}


def build_points_with_intensity() -> LiDARPoints:
    """Build a small cloud in front of the camera carrying an intensity."""
    return LiDARPoints(
        points=torch.tensor(
            [[0.0, 0.0, 2.0, 0.4], [0.5, 0.2, 4.0, 0.8], [-0.5, -0.2, 8.0, 0.2]],
            dtype=torch.float32,
        ),
        point_feature_names=[
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.INTENSITY,
        ],
        timestamp=0.0,
    )


def build_fusion_sample(camera_image_data: BaseImages) -> ModelGTSample:
    """Build a sample holding images and a small point cloud."""
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=build_points_with_intensity(),
        camera_image_data=camera_image_data,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )


class TestCalibrationMisalignment(unittest.TestCase):
    """Perturbation of the lidar to camera calibration."""

    def test_perturbs_the_calibration_and_labels_it(self) -> None:
        images = build_images()
        transform = CalibrationMisalignment(probability=1.0, **MISALIGNMENT_RANGES)

        sample = transform(build_sample(images))

        perturbed = sample.camera_image_data
        assert perturbed.calibration_statuses is not None
        self.assertEqual(
            perturbed.calibration_statuses.tolist(),
            [CalibrationStatus.MISCALIBRATED.value] * 2,
        )
        self.assertFalse(torch.equal(perturbed.lidar2cams, images.lidar2cams))

    def test_noise_composes_back_to_the_true_calibration(self) -> None:
        images = build_images()
        transform = CalibrationMisalignment(probability=1.0, **MISALIGNMENT_RANGES)

        sample = transform(build_sample(images))

        perturbed = sample.camera_image_data
        assert perturbed.noises is not None
        recovered = torch.linalg.inv(perturbed.noises) @ perturbed.lidar2cams
        torch.testing.assert_close(recovered, images.lidar2cams, atol=1e-5, rtol=0.0)

    def test_skipping_marks_the_sample_calibrated(self) -> None:
        images = build_images()
        transform = CalibrationMisalignment(probability=0.0, **MISALIGNMENT_RANGES)

        sample = transform(build_sample(images))

        calibrated = sample.camera_image_data
        assert calibrated.calibration_statuses is not None
        self.assertEqual(
            calibrated.calibration_statuses.tolist(), [CalibrationStatus.CALIBRATED.value] * 2
        )
        torch.testing.assert_close(calibrated.lidar2cams, images.lidar2cams)


class TestLidarCameraFusion(unittest.TestCase):
    """Projection of the points onto the images."""

    def test_projects_the_points_as_depth_and_intensity(self) -> None:
        images = build_images(height=32, width=32)
        transform = LidarCameraFusion(max_depth=128.0, dilation_size=0)

        sample = transform(build_fusion_sample(images))

        depth_maps = sample.camera_image_data.depth_maps
        assert depth_maps is not None
        self.assertEqual(depth_maps.shape, (2, 2, 32, 32))
        self.assertGreater(float(depth_maps[0, 0].max()), 0.0)
        self.assertGreater(float(depth_maps[0, 1].max()), 0.0)

    def test_a_perturbed_calibration_moves_the_projection(self) -> None:
        images = build_images(height=32, width=32)
        fusion = LidarCameraFusion(max_depth=128.0, dilation_size=0)
        misalignment = CalibrationMisalignment(probability=1.0, **MISALIGNMENT_RANGES)

        calibrated = fusion(build_fusion_sample(images))
        miscalibrated = fusion(misalignment(build_fusion_sample(images)))

        self.assertFalse(
            torch.equal(
                calibrated.camera_image_data.depth_maps,
                miscalibrated.camera_image_data.depth_maps,
            )
        )


class TestAffine(unittest.TestCase):
    """Affine warping of the images."""

    def test_records_the_affine_in_the_augmentation_matrices(self) -> None:
        transform = Affine(probability=1.0, max_distortion=0.1)

        sample = transform(build_sample(build_images(height=32, width=32)))

        affines = sample.camera_image_data.image_augmentation_pixel_affines()
        self.assertEqual(affines.shape, (2, 3, 3))
        self.assertFalse(torch.equal(affines, torch.eye(3).repeat(2, 1, 1)))

    def test_skipping_keeps_the_identity(self) -> None:
        transform = Affine(probability=0.0, max_distortion=0.1)

        sample = transform(build_sample(build_images(height=32, width=32)))

        affines = sample.camera_image_data.image_augmentation_pixel_affines()
        torch.testing.assert_close(affines, torch.eye(3).repeat(2, 1, 1))


class TestCropAndScale(unittest.TestCase):
    """Cropping and rescaling of the images."""

    def test_composes_the_crop_into_the_intrinsics_and_the_augmentation_matrices(self) -> None:
        # The pixel affine recorded in the augmentation matrices is the one the intrinsics
        # took, so the projection through the augmented intrinsics lands on the cropped pixels
        images = build_images(height=32, width=32)
        transform = CropAndScale(probability=1.0, crop_ratio=0.6)

        sample = transform(build_sample(images))

        cropped = sample.camera_image_data
        pixel_affines = cropped.augmented_camera_intrinsics @ torch.linalg.inv(
            images.augmented_camera_intrinsics
        )
        torch.testing.assert_close(cropped.image_augmentation_pixel_affines(), pixel_affines)
        self.assertFalse(
            torch.equal(cropped.augmented_camera_intrinsics, images.augmented_camera_intrinsics)
        )

    def test_keeps_the_image_size_and_scales_the_intrinsics(self) -> None:
        images = build_images(height=32, width=32)
        transform = CropAndScale(probability=1.0, crop_ratio=0.8)

        sample = transform(build_sample(images))

        cropped = sample.camera_image_data
        self.assertEqual(cropped.images.shape, images.images.shape)
        self.assertGreater(
            float(cropped.augmented_camera_intrinsics[0, 0, 0]),
            float(images.augmented_camera_intrinsics[0, 0, 0]),
        )


class TestCropBoxInner(unittest.TestCase):
    """Removal of the points inside a box."""

    def test_drops_the_points_inside_the_box(self) -> None:
        sample = build_fusion_sample(build_images())
        transform = CropBoxInner(crop_box=[-1.0, -1.0, 1.0, 1.0, 1.0, 5.0])

        cropped = transform(sample)

        self.assertEqual(cropped.point_cloud_data.coords[:, 2].tolist(), [8.0])


class TestSaveFusionPreview(unittest.TestCase):
    """Preview writing of the projected points."""

    def test_writes_a_depth_and_an_intensity_preview_per_camera(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            fused = LidarCameraFusion(max_depth=128.0, dilation_size=0)(
                build_fusion_sample(build_images(height=32, width=32))
            )

            SaveFusionPreview(out_dir=out_dir, probability=1.0)(fused)

            self.assertEqual(len(list(Path(out_dir).glob("*_depth.png"))), 2)
            self.assertEqual(len(list(Path(out_dir).glob("*_intensity.png"))), 2)

    def test_rejects_a_sample_without_the_projection(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            transform = SaveFusionPreview(out_dir=out_dir, probability=1.0)

            with self.assertRaisesRegex(ValueError, "no depth maps"):
                transform(build_sample(build_images()))
