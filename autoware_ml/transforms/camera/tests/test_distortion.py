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

"""Unit tests for the UndistortImage transform."""

import unittest

from jaxtyping import Float32
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.transforms.camera.distortion import UndistortImage


class TestUndistortImage(unittest.TestCase):
    """Unit tests for the UndistortImage transform."""

    def setUp(self) -> None:
        """Set up the camera calibration shared by all tests."""
        self.image_height = 48
        self.image_width = 64
        # A realistic pinhole intrinsic matrix for the image size under test.
        self.camera_intrinsic = torch.tensor(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.distortion_coefficients = torch.tensor(
            [0.1, -0.2, 0.001, 0.001, 0.05], dtype=torch.float32
        )

    def build_multi_task_gt_sample(
        self,
        distortion_coefficients: list[Float32[Tensor, " num_coefficients"]] | None = None,
        num_cameras: int = 2,
    ) -> ModelGTSample:
        """Build a minimal sample holding only camera image data.

        Args:
            distortion_coefficients: Distortion coefficients per camera. Defaults to the
                non-zero coefficients shared by the test case.
            num_cameras: Number of cameras the sample holds.

        Returns:
            ModelGTSample holding random images and the test camera calibration.
        """
        if distortion_coefficients is None:
            distortion_coefficients = [self.distortion_coefficients.clone()] * num_cameras

        camera_image_data = BaseImages(
            images=torch.rand(
                (num_cameras, 3, self.image_height, self.image_width), dtype=torch.float32
            ),
            timestamps=torch.zeros(num_cameras, dtype=torch.float64),
            camera_intrinsics=self.camera_intrinsic.repeat(num_cameras, 1, 1),
            camera_names=[f"camera{index}" for index in range(num_cameras)],
            lidar2images=torch.eye(4).repeat(num_cameras, 1, 1),
            lidar2cams=torch.eye(4).repeat(num_cameras, 1, 1),
            distortion_models=["plumb_bob"] * num_cameras,
            distortion_coefficients=distortion_coefficients,
            augmented_camera_intrinsics=self.camera_intrinsic.repeat(num_cameras, 1, 1),
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

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom alpha."""
        self.assertEqual(UndistortImage().alpha, 0.0)
        self.assertEqual(UndistortImage(alpha=0.5).alpha, 0.5)

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            UndistortImage()(sample)

    def test_passthrough_zero_distortion(self) -> None:
        """Test that zero distortion coefficients pass the images through unchanged."""
        sample = self.build_multi_task_gt_sample(
            distortion_coefficients=[torch.zeros(5), torch.zeros(5)]
        )
        assert sample.camera_image_data is not None

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        self.assertTrue(
            torch.equal(output.camera_image_data.images, sample.camera_image_data.images)
        )
        self.assertTrue(
            torch.equal(
                output.camera_image_data.augmented_camera_intrinsics,
                sample.camera_image_data.augmented_camera_intrinsics,
            )
        )

    def test_output_shape_preserved(self) -> None:
        """Test that the undistorted images keep the input shape and dtype."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        images = sample.camera_image_data.images

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        self.assertEqual(output.camera_image_data.images.shape, images.shape)
        self.assertEqual(output.camera_image_data.images.dtype, images.dtype)
        # Undistortion of non-zero coefficients has to change the pixels.
        self.assertFalse(torch.equal(output.camera_image_data.images, images))

    def test_new_camera_matrix_updated(self) -> None:
        """Test that the augmented intrinsics are replaced and the coefficients cleared."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        for index in range(len(camera_image_data.camera_names)):
            self.assertFalse(
                torch.allclose(
                    camera_image_data.augmented_camera_intrinsics[index], self.camera_intrinsic
                )
            )
            # The coefficients are cleared, the images are pinhole now.
            self.assertEqual(camera_image_data.distortion_coefficients[index].numel(), 0)
        # The raw intrinsics stay untouched, only the augmented ones follow the transform.
        self.assertTrue(
            torch.equal(
                camera_image_data.camera_intrinsics, sample.camera_image_data.camera_intrinsics
            )
        )

    def test_camera_image_data_returned(self) -> None:
        """Test that the output holds camera image data carrying over the untouched fields."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertIsInstance(camera_image_data, BaseImages)
        self.assertEqual(camera_image_data.camera_names, sample.camera_image_data.camera_names)
        self.assertEqual(
            camera_image_data.distortion_models, [""] * len(camera_image_data.camera_names)
        )
        self.assertTrue(
            torch.equal(camera_image_data.timestamps, sample.camera_image_data.timestamps)
        )
        # The projections follow the new intrinsics.
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images,
                compose_lidar2images(
                    camera_image_data.augmented_camera_intrinsics, camera_image_data.lidar2cams
                ),
            )
        )
        self.assertTrue(
            torch.equal(camera_image_data.lidar2cams, sample.camera_image_data.lidar2cams)
        )
        # The undistortion is not a 2D affine, it is carried by the intrinsics alone.
        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                sample.camera_image_data.image_augmentation_matrices,
            )
        )

    def test_applying_twice_is_a_no_op(self) -> None:
        """Test that a second pass leaves the already undistorted images unchanged."""
        transform = UndistortImage()
        undistorted = transform(self.build_multi_task_gt_sample())
        assert undistorted.camera_image_data is not None

        output = transform(undistorted)

        assert output.camera_image_data is not None
        self.assertTrue(
            torch.equal(output.camera_image_data.images, undistorted.camera_image_data.images)
        )
        self.assertTrue(
            torch.equal(
                output.camera_image_data.augmented_camera_intrinsics,
                undistorted.camera_image_data.augmented_camera_intrinsics,
            )
        )


if __name__ == "__main__":
    unittest.main()
