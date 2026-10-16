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

"""Unit tests for the UndistortImage transform.

The images under test are synthetic: a single bright Gaussian feature per camera rendered
where a lens with known coefficients images a known 3D point, on a black background and at
an asymmetric location. The transform output is then compared against an OpenCV reference
computed in the test, and the feature is required to land where the pinhole camera of the
new intrinsics projects the 3D point. An identity or a flipped image fails both checks.
"""

import unittest
from typing import Any

import cv2
from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.databases.schemas.image_frames import distortion_model_from_coefficients
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.transforms.camera.distortion import UndistortImage
from autoware_ml.transforms.camera.resize import ResizeMultiviewImages


class TestUndistortImage(unittest.TestCase):
    """Unit tests for the UndistortImage transform."""

    def setUp(self) -> None:
        """Set up the camera calibration and the synthetic scene shared by all tests."""
        self.image_height = 96
        self.image_width = 128
        self.num_cameras = 2
        # An off-centre principal point and a strong barrel distortion, so the lens moves
        # the features by several pixels and no symmetric pixel shuffle can mimic it.
        self.camera_intrinsic = torch.tensor(
            [[80.0, 0.0, 70.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.distortion_coefficients = torch.tensor(
            [-0.25, 0.08, 0.002, -0.003, 0.0], dtype=torch.float32
        )
        # One 3D point per camera in normalized camera coordinates (unit depth), placed on
        # different sides of the principal point so the two images are not mirrors.
        self.feature_points = torch.tensor(
            [[0.35, -0.2, 1.0], [-0.3, 0.3, 1.0]], dtype=torch.float64
        )
        # Gaussian feature width in pixels, as twice the squared standard deviation.
        self.feature_two_sigma_squared = 8.0
        # Pixel tolerance on the feature centroid after bilinear resampling.
        self.centroid_tolerance = 0.25

    # ------------------------------------------------------------------ reference helpers

    def camera_matrix(self) -> np.ndarray:
        """Return the camera matrix the way the transform hands it to OpenCV.

        Returns:
            Float64 (3, 3) camera matrix.
        """
        return self.camera_intrinsic.numpy().astype(np.float64)

    def coefficients(self) -> np.ndarray:
        """Return the distortion coefficients the way the transform hands them to OpenCV.

        Returns:
            Float64 (5,) distortion coefficients.
        """
        return self.distortion_coefficients.numpy().astype(np.float64)

    def reference_new_camera_matrix(self, alpha: float) -> np.ndarray:
        """Compute the optimal new camera matrix OpenCV picks for the test calibration.

        Args:
            alpha: Free scaling parameter of the undistortion.

        Returns:
            Float64 (3, 3) pinhole camera matrix of the undistorted image.
        """
        image_size = (self.image_width, self.image_height)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix(), self.coefficients(), image_size, alpha, image_size
        )
        return new_camera_matrix

    def reference_undistort(
        self, image: Float32[Tensor, "num_channels height width"], alpha: float
    ) -> Float32[Tensor, "num_channels height width"]:
        """Undistort one image with OpenCV directly.

        Args:
            image: Distorted image in (num_channels, height, width) layout.
            alpha: Free scaling parameter of the undistortion.

        Returns:
            Undistorted image in (num_channels, height, width) layout.
        """
        undistorted = cv2.undistort(
            image.permute(1, 2, 0).contiguous().numpy(),
            self.camera_matrix(),
            self.coefficients(),
            newCameraMatrix=self.reference_new_camera_matrix(alpha),
        )
        return torch.from_numpy(undistorted).permute(2, 0, 1)

    def distorted_feature_pixels(self) -> np.ndarray:
        """Project the feature points through the distorting lens.

        Returns:
            Float64 (num_cameras, 2) pixel location of each camera's feature in its raw image.
        """
        pixels, _ = cv2.projectPoints(
            self.feature_points.numpy(),
            np.zeros(3),
            np.zeros(3),
            self.camera_matrix(),
            self.coefficients(),
        )
        return pixels.reshape(-1, 2)

    def pinhole_feature_pixels(self, camera_matrix: np.ndarray) -> np.ndarray:
        """Project the feature points through a pinhole camera.

        Args:
            camera_matrix: Float64 (3, 3) pinhole camera matrix.

        Returns:
            Float64 (num_cameras, 2) pixel location of each camera's feature.
        """
        homogeneous = self.feature_points.numpy() @ camera_matrix.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    def render_feature_image(
        self, center: np.ndarray
    ) -> Float32[Tensor, "num_channels height width"]:
        """Render a Gaussian feature on a black image, with a different gain per channel.

        Args:
            center: (2,) sub-pixel (x, y) location of the feature.

        Returns:
            Float32 (3, height, width) unit-range image.
        """
        rows, columns = torch.meshgrid(
            torch.arange(self.image_height, dtype=torch.float32),
            torch.arange(self.image_width, dtype=torch.float32),
            indexing="ij",
        )
        squared_distance = (columns - float(center[0])).square() + (
            rows - float(center[1])
        ).square()
        feature = torch.exp(-squared_distance / self.feature_two_sigma_squared)
        return torch.stack([feature, feature / 2.0, feature / 4.0], dim=0)

    def build_feature_images(self) -> Float32[Tensor, "num_cameras num_channels height width"]:
        """Render the raw images of the distorting cameras, one feature per camera.

        Returns:
            Float32 (num_cameras, 3, height, width) images.
        """
        return torch.stack(
            [self.render_feature_image(center) for center in self.distorted_feature_pixels()],
            dim=0,
        )

    def feature_centroid(self, channel: Float32[Tensor, "height width"]) -> np.ndarray:
        """Locate a feature as the intensity-weighted centroid of a channel.

        Args:
            channel: Single-channel image holding one feature on a black background.

        Returns:
            Float64 (2,) sub-pixel (x, y) centroid.
        """
        rows, columns = torch.meshgrid(
            torch.arange(self.image_height, dtype=torch.float32),
            torch.arange(self.image_width, dtype=torch.float32),
            indexing="ij",
        )
        mass = channel.sum()
        return np.array(
            [
                float(torch.mul(columns, channel).sum() / mass),
                float(torch.mul(rows, channel).sum() / mass),
            ]
        )

    # -------------------------------------------------------------------- sample builders

    def build_camera_image_fields(
        self,
        images: Float32[Tensor, "num_cameras num_channels height width"] | None = None,
        distortion_coefficients: list[Float32[Tensor, " num_coefficients"]] | None = None,
    ) -> dict[str, Any]:
        """Build the fields of a BaseImages holding the test cameras.

        Args:
            images: Images per camera. Defaults to the synthetic feature images.
            distortion_coefficients: Coefficients per camera. Defaults to the shared non-zero
                coefficients.

        Returns:
            Field dictionary accepted by ``BaseImages.model_validate``.
        """
        if images is None:
            images = self.build_feature_images()
        if distortion_coefficients is None:
            distortion_coefficients = [
                self.distortion_coefficients.clone() for _ in range(self.num_cameras)
            ]
        camera_intrinsics = self.camera_intrinsic.repeat(self.num_cameras, 1, 1)
        lidar2cams = torch.eye(4).repeat(self.num_cameras, 1, 1)
        return {
            "images": images,
            "timestamps": torch.zeros(self.num_cameras, dtype=torch.float64),
            "camera_intrinsics": camera_intrinsics,
            "camera_names": [f"camera{index}" for index in range(self.num_cameras)],
            "lidar2images": compose_lidar2images(camera_intrinsics, lidar2cams),
            "lidar2cams": lidar2cams,
            "distortion_models": [
                distortion_model_from_coefficients(coefficients.tolist())
                for coefficients in distortion_coefficients
            ],
            "distortion_coefficients": distortion_coefficients,
            "augmented_camera_intrinsics": camera_intrinsics.clone(),
            "image_augmentation_matrices": torch.eye(4).repeat(self.num_cameras, 1, 1),
        }

    def build_multi_task_gt_sample(
        self,
        images: Float32[Tensor, "num_cameras num_channels height width"] | None = None,
        distortion_coefficients: list[Float32[Tensor, " num_coefficients"]] | None = None,
    ) -> ModelGTSample:
        """Build a minimal sample holding only camera image data.

        Args:
            images: Images per camera. Defaults to the synthetic feature images.
            distortion_coefficients: Coefficients per camera. Defaults to the shared non-zero
                coefficients.

        Returns:
            ModelGTSample holding the images and the test camera calibration.
        """
        camera_image_data = BaseImages.model_validate(
            self.build_camera_image_fields(images, distortion_coefficients)
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    # ------------------------------------------------------------------------------ tests

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom alpha."""
        self.assertEqual(UndistortImage().alpha, 0.0)
        self.assertEqual(UndistortImage(alpha=0.5).alpha, 0.5)

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            UndistortImage()(sample)

    def test_scene_is_discriminating(self) -> None:
        """Test that the synthetic scene tells undistortion apart from identity and flips."""
        distorted = self.distorted_feature_pixels()
        pinhole = self.pinhole_feature_pixels(self.reference_new_camera_matrix(alpha=0.0))
        last_column = self.image_width - 1
        last_row = self.image_height - 1
        for index in range(self.num_cameras):
            with self.subTest(camera=index):
                # The lens moves the feature by more than the centroid tolerance.
                self.assertGreater(np.linalg.norm(distorted[index] - pinhole[index]), 1.0)
                # Neither flip of the raw image puts the feature at its pinhole location.
                horizontal_flip = np.array([last_column - distorted[index, 0], distorted[index, 1]])
                vertical_flip = np.array([distorted[index, 0], last_row - distorted[index, 1]])
                self.assertGreater(np.linalg.norm(horizontal_flip - pinhole[index]), 1.0)
                self.assertGreater(np.linalg.norm(vertical_flip - pinhole[index]), 1.0)
            # The feature stays inside the undistorted image, so its centroid is measurable.
            self.assertTrue(np.all(pinhole[index] > 4.0))
            self.assertLess(pinhole[index, 0], last_column - 4.0)
            self.assertLess(pinhole[index, 1], last_row - 4.0)

    def test_matches_opencv_reference(self) -> None:
        """Test that the images match cv2.undistort run with the same calibration."""
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                sample = self.build_multi_task_gt_sample()
                assert sample.camera_image_data is not None
                images = sample.camera_image_data.images

                output = UndistortImage(alpha=alpha)(sample)

                assert output.camera_image_data is not None
                for index in range(self.num_cameras):
                    expected = self.reference_undistort(images[index], alpha)
                    self.assertTrue(
                        torch.allclose(output.camera_image_data.images[index], expected, atol=1e-5)
                    )

    def test_feature_lands_at_pinhole_projection(self) -> None:
        """Test that each feature moves to where the new pinhole intrinsics project it."""
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                new_camera_matrix = self.reference_new_camera_matrix(alpha)
                pinhole = self.pinhole_feature_pixels(new_camera_matrix)
                distorted = self.distorted_feature_pixels()
                sample = self.build_multi_task_gt_sample()

                output = UndistortImage(alpha=alpha)(sample)

                assert output.camera_image_data is not None
                images = output.camera_image_data.images
                for index in range(self.num_cameras):
                    centroid = self.feature_centroid(images[index, 0])
                    self.assertLess(
                        np.linalg.norm(centroid - pinhole[index]), self.centroid_tolerance
                    )
                    # The feature really moved away from where the lens imaged it.
                    self.assertGreater(np.linalg.norm(centroid - distorted[index]), 1.0)
                    # The per-channel gains survive the resampling, the channels are not mixed.
                    peak = images[index, 0].max()
                    self.assertGreater(float(peak), 0.8)
                    self.assertTrue(torch.allclose(images[index, 1], images[index, 0] / 2.0))
                    self.assertTrue(torch.allclose(images[index, 2], images[index, 0] / 4.0))

    def test_output_is_not_a_flip(self) -> None:
        """Test that the output is neither the input nor a flipped copy of it."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        images = sample.camera_image_data.images

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        undistorted = output.camera_image_data.images
        for candidate in (images, images.flip(dims=[3]), images.flip(dims=[2])):
            self.assertGreater(float((undistorted - candidate).abs().max()), 0.5)

    def test_output_shape_preserved(self) -> None:
        """Test that the undistorted images keep the input shape and dtype."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        images = sample.camera_image_data.images

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        self.assertEqual(output.camera_image_data.images.shape, images.shape)
        self.assertEqual(output.camera_image_data.images.dtype, images.dtype)

    def test_passthrough_zero_distortion(self) -> None:
        """Test that all-zero coefficients pass the images and intrinsics through unchanged."""
        images = torch.rand(
            (self.num_cameras, 3, self.image_height, self.image_width), dtype=torch.float32
        )
        sample = self.build_multi_task_gt_sample(
            images=images, distortion_coefficients=[torch.zeros(5), torch.zeros(5)]
        )
        assert sample.camera_image_data is not None

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertTrue(torch.equal(camera_image_data.images, images))
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                sample.camera_image_data.augmented_camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(camera_image_data.lidar2images, sample.camera_image_data.lidar2images)
        )
        # A zero plumb_bob is a pinhole, it is labelled as pre-undistorted on the way out.
        for index in range(self.num_cameras):
            self.assertEqual(camera_image_data.distortion_coefficients[index].numel(), 0)
            self.assertEqual(camera_image_data.distortion_models[index], "")

    def test_passthrough_empty_coefficients(self) -> None:
        """Test that pre-undistorted cameras without coefficients pass through unchanged."""
        images = torch.rand(
            (self.num_cameras, 3, self.image_height, self.image_width), dtype=torch.float32
        )
        sample = self.build_multi_task_gt_sample(
            images=images, distortion_coefficients=[torch.zeros(0), torch.zeros(0)]
        )
        assert sample.camera_image_data is not None
        self.assertEqual(sample.camera_image_data.distortion_models, ["", ""])

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertTrue(torch.equal(camera_image_data.images, images))
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                sample.camera_image_data.augmented_camera_intrinsics,
            )
        )
        for index in range(self.num_cameras):
            self.assertEqual(camera_image_data.distortion_coefficients[index].numel(), 0)

    def test_mixed_cameras_only_distorted_ones_change(self) -> None:
        """Test that a pinhole camera next to a distorting one is left alone."""
        images = self.build_feature_images()
        sample = self.build_multi_task_gt_sample(
            images=images,
            distortion_coefficients=[self.distortion_coefficients.clone(), torch.zeros(5)],
        )
        assert sample.camera_image_data is not None

        output = UndistortImage()(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertTrue(
            torch.allclose(
                camera_image_data.images[0],
                self.reference_undistort(images[0], alpha=0.0),
                atol=1e-5,
            )
        )
        self.assertTrue(torch.equal(camera_image_data.images[1], images[1]))
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics[0],
                torch.from_numpy(self.reference_new_camera_matrix(alpha=0.0)).to(torch.float32),
                atol=1e-4,
            )
        )
        self.assertTrue(
            torch.equal(camera_image_data.augmented_camera_intrinsics[1], self.camera_intrinsic)
        )

    def test_new_camera_matrix_updated(self) -> None:
        """Test that the augmented intrinsics become the optimal new camera matrix."""
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                sample = self.build_multi_task_gt_sample()
                assert sample.camera_image_data is not None
                expected = torch.from_numpy(self.reference_new_camera_matrix(alpha)).to(
                    torch.float32
                )

                output = UndistortImage(alpha=alpha)(sample)

                assert output.camera_image_data is not None
                camera_image_data = output.camera_image_data
                for index in range(self.num_cameras):
                    self.assertTrue(
                        torch.allclose(
                            camera_image_data.augmented_camera_intrinsics[index],
                            expected,
                            atol=1e-4,
                        )
                    )
                    self.assertFalse(
                        torch.allclose(
                            camera_image_data.augmented_camera_intrinsics[index],
                            self.camera_intrinsic,
                        )
                    )
                    # The coefficients are cleared and the model name agrees with them.
                    self.assertEqual(camera_image_data.distortion_coefficients[index].numel(), 0)
                    self.assertEqual(
                        camera_image_data.distortion_models[index],
                        distortion_model_from_coefficients(
                            camera_image_data.distortion_coefficients[index].tolist()
                        ),
                    )
                # The raw intrinsics stay untouched, only the augmented ones follow.
                self.assertTrue(
                    torch.equal(
                        camera_image_data.camera_intrinsics,
                        sample.camera_image_data.camera_intrinsics,
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
        self.assertEqual(camera_image_data.distortion_models, ["", ""])
        self.assertTrue(
            torch.equal(camera_image_data.timestamps, sample.camera_image_data.timestamps)
        )
        # The projections follow the new intrinsics and no longer match the raw ones.
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images,
                compose_lidar2images(
                    camera_image_data.augmented_camera_intrinsics, camera_image_data.lidar2cams
                ),
            )
        )
        self.assertFalse(
            torch.allclose(camera_image_data.lidar2images, sample.camera_image_data.lidar2images)
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

    def test_raises_after_an_image_space_transform(self) -> None:
        """
        Test that undistorting an already resized image is rejected: the coefficients describe
        the raw image, so the transform would otherwise write raw-frame intrinsics over the
        augmented ones and report a principal point outside the resized image.
        """
        resized = ResizeMultiviewImages(
            target_size=[self.image_height // 2, self.image_width // 2]
        )(self.build_multi_task_gt_sample())

        with self.assertRaisesRegex(ValueError, "before any image-space transform"):
            UndistortImage()(resized)

    def test_undistorted_cameras_pass_through_after_an_image_space_transform(self) -> None:
        """
        Test that the ordering check only guards cameras that still have to be undistorted,
        so a second pass over already undistorted and then resized images stays a no-op.
        """
        transform = UndistortImage()
        resized = ResizeMultiviewImages(
            target_size=[self.image_height // 2, self.image_width // 2]
        )(transform(self.build_multi_task_gt_sample()))
        assert resized.camera_image_data is not None

        output = transform(resized)

        assert output.camera_image_data is not None
        self.assertTrue(
            torch.equal(output.camera_image_data.images, resized.camera_image_data.images)
        )
        self.assertTrue(
            torch.equal(
                output.camera_image_data.augmented_camera_intrinsics,
                resized.camera_image_data.augmented_camera_intrinsics,
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
        self.assertTrue(
            torch.equal(
                output.camera_image_data.lidar2images, undistorted.camera_image_data.lidar2images
            )
        )


if __name__ == "__main__":
    unittest.main()
