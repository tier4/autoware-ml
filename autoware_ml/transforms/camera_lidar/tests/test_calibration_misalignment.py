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

"""Unit tests for the CalibrationMisalignment transform."""

from __future__ import annotations

import math
import unittest

from jaxtyping import Float32
from pydantic import ValidationError
import torch
from torch import Tensor
import transforms3d

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera_lidar.camera_lidar import (
    AxisActivation,
    CalibrationMisalignment,
    MagnitudeRange,
    TranslationRange,
)
from autoware_ml.utils.calibration import CalibrationStatus


def _build_sample(num_cameras: int) -> ModelGTSample:
    """Build a sample with ``num_cameras`` pinhole cameras at distinct poses."""
    camera_intrinsics = torch.tensor(
        [[10.0, 0.0, 6.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    ).repeat(num_cameras, 1, 1)
    lidar2cams = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
    # Give every camera its own translation so per-camera effects are distinguishable.
    lidar2cams[:, :3, 3] = torch.arange(num_cameras, dtype=torch.float32)[:, None] + 1.0
    homogeneous_intrinsics = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
    homogeneous_intrinsics[:, :3, :3] = camera_intrinsics

    camera_image_data = BaseImages(
        images=torch.ones((num_cameras, 3, 8, 12), dtype=torch.float32),
        timestamps=torch.zeros(num_cameras, dtype=torch.float64),
        camera_intrinsics=camera_intrinsics,
        camera_names=[f"camera{index}" for index in range(num_cameras)],
        lidar2images=homogeneous_intrinsics @ lidar2cams,
        lidar2cams=lidar2cams,
        distortion_models=["plumb_bob"] * num_cameras,
        distortion_coefficients=[torch.zeros(5) for _ in range(num_cameras)],
        augmented_camera_intrinsics=camera_intrinsics.clone(),
        image_augmentation_matrices=torch.eye(4).repeat(num_cameras, 1, 1),
    )
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=camera_image_data,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )


def _homogeneous(
    intrinsics: Float32[Tensor, "num_cameras 3 3"],
) -> Float32[Tensor, "num_cameras 4 4"]:
    """Pad 3x3 intrinsics to 4x4 homogeneous matrices."""
    homogeneous = torch.eye(4, dtype=intrinsics.dtype).repeat(intrinsics.shape[0], 1, 1)
    homogeneous[:, :3, :3] = intrinsics
    return homogeneous


class TestCalibrationMisalignmentInit(unittest.TestCase):
    """Constructor defaults and pydantic validation of the parameter groups."""

    def test_instantiation_defaults(self) -> None:
        transform = CalibrationMisalignment(probability=0.5)

        self.assertEqual(transform.activate, AxisActivation())
        self.assertEqual(transform.roll, MagnitudeRange())
        self.assertEqual(transform.pitch, MagnitudeRange())
        self.assertEqual(transform.yaw, MagnitudeRange())
        self.assertEqual(transform.translation, TranslationRange())
        for axis in ("x", "y", "z"):
            self.assertEqual(transform.translation.axis(axis), MagnitudeRange())

    def test_instantiation_single_axis(self) -> None:
        transform = CalibrationMisalignment(
            probability=0.5,
            activate=AxisActivation(roll=True),
            roll=MagnitudeRange(min_neg=1.0, max_neg=5.0, min_pos=2.0, max_pos=6.0),
        )

        self.assertTrue(transform.activate.roll)
        self.assertFalse(transform.activate.pitch)
        self.assertEqual(
            transform.roll,
            MagnitudeRange(min_neg=1.0, max_neg=5.0, min_pos=2.0, max_pos=6.0),
        )
        self.assertEqual(transform.pitch, MagnitudeRange())

    def test_instantiation_translation_group(self) -> None:
        transform = CalibrationMisalignment(
            probability=0.5,
            activate=AxisActivation(x=True, z=True),
            translation=TranslationRange(
                min_x_neg=0.1,
                max_x_neg=0.5,
                min_x_pos=0.2,
                max_x_pos=0.6,
                min_z_neg=0.3,
                max_z_neg=0.7,
            ),
        )

        self.assertTrue(transform.activate.x)
        self.assertFalse(transform.activate.y)
        self.assertTrue(transform.activate.z)
        self.assertEqual(
            transform.translation.axis("x"),
            MagnitudeRange(min_neg=0.1, max_neg=0.5, min_pos=0.2, max_pos=0.6),
        )
        self.assertEqual(transform.translation.axis("y"), MagnitudeRange())
        self.assertEqual(transform.translation.axis("z"), MagnitudeRange(min_neg=0.3, max_neg=0.7))

    def test_instantiation_accepts_model_instances(self) -> None:
        transform = CalibrationMisalignment(
            probability=0.5,
            activate=AxisActivation(yaw=True),
            yaw=MagnitudeRange(min_neg=1.0, max_neg=2.0),
            translation=TranslationRange(min_y_pos=0.1, max_y_pos=0.2),
        )

        self.assertTrue(transform.activate.yaw)
        self.assertEqual(transform.yaw, MagnitudeRange(min_neg=1.0, max_neg=2.0))
        self.assertEqual(transform.translation.axis("y"), MagnitudeRange(min_pos=0.1, max_pos=0.2))

    def test_validation_negative_magnitude(self) -> None:
        with self.assertRaisesRegex(ValidationError, "greater than or equal to 0"):
            CalibrationMisalignment(probability=0.5, roll=MagnitudeRange(min_neg=-1.0))

        with self.assertRaisesRegex(ValidationError, "greater than or equal to 0"):
            CalibrationMisalignment(probability=0.5, translation=TranslationRange(max_z_pos=-1.0))

    def test_validation_min_greater_than_max(self) -> None:
        with self.assertRaisesRegex(ValidationError, r"min_neg \(5.0\) must be <= max_neg \(1.0\)"):
            CalibrationMisalignment(probability=0.5, roll=MagnitudeRange(min_neg=5.0, max_neg=1.0))

        with self.assertRaisesRegex(
            ValidationError, r"min_y_pos \(0.5\) must be <= max_y_pos \(0.1\)"
        ):
            CalibrationMisalignment(
                probability=0.5, translation=TranslationRange(min_y_pos=0.5, max_y_pos=0.1)
            )

    def test_validation_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValidationError, "Extra inputs are not permitted"):
            CalibrationMisalignment(probability=0.5, activate=AxisActivation(rol=True))

        with self.assertRaisesRegex(ValidationError, "Extra inputs are not permitted"):
            CalibrationMisalignment(probability=0.5, translation=TranslationRange(min_roll_neg=1.0))

    def test_config_groups_are_frozen(self) -> None:
        transform = CalibrationMisalignment(probability=0.5)

        with self.assertRaises(ValidationError):
            transform.activate.roll = True  # type: ignore[misc]

    def test_missing_camera_image_data(self) -> None:
        transform = CalibrationMisalignment(probability=0.5)
        sample = _build_sample(1)._replace(camera_image_data=None)

        with self.assertRaisesRegex(KeyError, "Missing required key 'camera_image_data'"):
            transform(sample)


class TestCalibrationMisalignmentTransform(unittest.TestCase):
    """Behavior of the transform on a ModelGTSample."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.sample = _build_sample(num_cameras=2)
        self.roll_transform = CalibrationMisalignment(
            probability=1.0,
            activate=AxisActivation(roll=True),
            roll=MagnitudeRange(min_neg=5.0, max_neg=10.0, min_pos=5.0, max_pos=10.0),
        )

    def test_probability_zero_marks_calibrated(self) -> None:
        transform = CalibrationMisalignment(probability=0.0, activate=AxisActivation(roll=True))
        original = self.sample.camera_image_data

        output = transform(self.sample).camera_image_data

        torch.testing.assert_close(output.lidar2cams, original.lidar2cams)
        torch.testing.assert_close(output.lidar2images, original.lidar2images)
        self.assertIsNone(output.noises)
        self.assertTrue(
            torch.equal(
                output.calibration_statuses,
                torch.full((2,), CalibrationStatus.CALIBRATED.value, dtype=torch.int64),
            )
        )

    def test_probability_one_perturbs_extrinsics(self) -> None:
        original = self.sample.camera_image_data

        output = self.roll_transform(self.sample).camera_image_data

        self.assertFalse(torch.allclose(output.lidar2cams, original.lidar2cams))
        self.assertEqual(tuple(output.noises.shape), (2, 4, 4))
        self.assertTrue(
            torch.equal(
                output.calibration_statuses,
                torch.full((2,), CalibrationStatus.MISCALIBRATED.value, dtype=torch.int64),
            )
        )

    def test_noise_is_applied_in_camera_frame(self) -> None:
        original = self.sample.camera_image_data

        output = self.roll_transform(self.sample).camera_image_data

        # T_noisy = T_noise @ T_lidar2cam, per camera.
        torch.testing.assert_close(output.lidar2cams, output.noises @ original.lidar2cams)

    def test_lidar2images_follow_perturbed_extrinsics(self) -> None:
        output = self.roll_transform(self.sample).camera_image_data

        expected = _homogeneous(output.augmented_camera_intrinsics) @ output.lidar2cams
        torch.testing.assert_close(output.lidar2images, expected)

    def test_each_camera_draws_its_own_noise(self) -> None:
        output = self.roll_transform(self.sample).camera_image_data

        self.assertFalse(torch.allclose(output.noises[0], output.noises[1]))

    def test_no_active_axes_gives_identity_noise_but_miscalibrated(self) -> None:
        transform = CalibrationMisalignment(probability=1.0)
        original = self.sample.camera_image_data

        output = transform(self.sample).camera_image_data

        torch.testing.assert_close(output.lidar2cams, original.lidar2cams)
        torch.testing.assert_close(output.noises, torch.eye(4).repeat(2, 1, 1))
        self.assertTrue(
            torch.equal(
                output.calibration_statuses,
                torch.full((2,), CalibrationStatus.MISCALIBRATED.value, dtype=torch.int64),
            )
        )

    def test_input_sample_is_not_mutated(self) -> None:
        original_lidar2cams = self.sample.camera_image_data.lidar2cams.clone()

        self.roll_transform(self.sample)

        torch.testing.assert_close(self.sample.camera_image_data.lidar2cams, original_lidar2cams)
        self.assertIsNone(self.sample.camera_image_data.calibration_statuses)


class TestCalibrationMisalignmentSampling(unittest.TestCase):
    """Noise sampling helpers."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.transform = CalibrationMisalignment(probability=0.5)

    def test_rotation_matrix_matches_transforms3d_sxyz(self) -> None:
        roll, pitch, yaw = 0.3, -0.2, 0.7

        rotation = CalibrationMisalignment.rotation_matrix(roll, pitch, yaw)

        expected = torch.tensor(
            transforms3d.euler.euler2mat(roll, pitch, yaw, axes="sxyz"), dtype=torch.float32
        )
        torch.testing.assert_close(rotation, expected)

    def test_sample_noise_transform_translation_only(self) -> None:
        transform = CalibrationMisalignment(
            probability=1.0,
            activate=AxisActivation(x=True),
            translation=TranslationRange(
                min_x_neg=0.5, max_x_neg=0.5, min_x_pos=0.5, max_x_pos=0.5
            ),
        )

        noise = transform.sample_noise_transform()

        torch.testing.assert_close(noise[:3, :3], torch.eye(3))
        self.assertAlmostEqual(abs(float(noise[0, 3])), 0.5)
        self.assertEqual(float(noise[1, 3]), 0.0)
        self.assertEqual(float(noise[2, 3]), 0.0)

    def test_sample_component_respects_sign_and_range(self) -> None:
        bounds = MagnitudeRange(min_neg=1.0, max_neg=2.0, min_pos=3.0, max_pos=4.0)
        values = [self.transform._sample_component(bounds) for _ in range(200)]

        negatives = [value for value in values if value < 0]
        positives = [value for value in values if value > 0]
        self.assertTrue(negatives and positives)
        self.assertTrue(all(-2.0 <= value <= -1.0 for value in negatives))
        self.assertTrue(all(3.0 <= value <= 4.0 for value in positives))

    def test_alter_calibration_shape(self) -> None:
        transform = CalibrationMisalignment(
            probability=1.0,
            activate=AxisActivation(roll=True),
            roll=MagnitudeRange(min_neg=1.0, max_neg=5.0, min_pos=1.0, max_pos=5.0),
        )
        lidar2cam = torch.eye(4)

        noisy, noise = transform.alter_calibration(lidar2cam)

        self.assertEqual(tuple(noisy.shape), (4, 4))
        self.assertEqual(tuple(noise.shape), (4, 4))
        torch.testing.assert_close(noisy, noise @ lidar2cam)
        # Only roll is active, so the noise is a rotation about x within the configured range.
        roll_degrees = math.degrees(math.atan2(float(noise[2, 1]), float(noise[2, 2])))
        self.assertTrue(1.0 <= abs(roll_degrees) <= 5.0)
        self.assertEqual(float(noise[0, 3]), 0.0)

    def test_alter_calibration_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Transform must be 4x4 matrix"):
            self.transform.alter_calibration(torch.eye(3))

    def test_bounded_gaussian_values_in_range(self) -> None:
        samples = [self.transform.bounded_gaussian(2.0, 1.0, 5.0, 1.0) for _ in range(100)]

        self.assertTrue(all(1.0 <= sample <= 5.0 for sample in samples))

    def test_bounded_gaussian_invalid_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_value .* must be less than max_value"):
            self.transform.bounded_gaussian(center=1.0, min_value=5.0, max_value=1.0, scale=1.0)

    def test_bounded_gaussian_invalid_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "scale .* must be positive"):
            self.transform.bounded_gaussian(center=1.0, min_value=0.0, max_value=5.0, scale=0.0)


if __name__ == "__main__":
    unittest.main()
