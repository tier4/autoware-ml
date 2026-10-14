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

"""Unit tests for the camera image resize, crop, and padding transforms."""

from typing import Tuple
import unittest

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.resize import (
    CropAndScale,
    PadMultiViewImage,
    ResizeCropFlipRotImage,
    ResizeMultiviewImages,
)


class CameraImageDataTestCase(unittest.TestCase):
    """Base test case building a sample holding only camera image data."""

    def setUp(self) -> None:
        """Set up the camera calibration shared by all tests."""
        self.image_height = 32
        self.image_width = 48
        self.num_cameras = 2
        # A realistic pinhole intrinsic matrix for the image size under test.
        self.camera_intrinsic = torch.tensor(
            [[100.0, 0.0, 24.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        # A camera looking forward, one metre above and behind the lidar.
        self.lidar2cam = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def build_multi_task_gt_sample(self) -> ModelGTSample:
        """Build a minimal sample holding only camera image data.

        Returns:
            ModelGTSample holding random images and the test camera calibration.
        """
        homogeneous_intrinsic = torch.eye(4, dtype=torch.float32)
        homogeneous_intrinsic[:3, :3] = self.camera_intrinsic

        camera_image_data = BaseImages(
            images=torch.rand(
                (self.num_cameras, 3, self.image_height, self.image_width), dtype=torch.float32
            ),
            timestamps=torch.zeros(self.num_cameras, dtype=torch.float64),
            camera_intrinsics=self.camera_intrinsic.repeat(self.num_cameras, 1, 1),
            camera_names=[f"camera{index}" for index in range(self.num_cameras)],
            lidar2images=(homogeneous_intrinsic @ self.lidar2cam).repeat(self.num_cameras, 1, 1),
            lidar2cams=self.lidar2cam.repeat(self.num_cameras, 1, 1),
            distortion_models=["plumb_bob"] * self.num_cameras,
            distortion_coefficients=[torch.zeros(5)] * self.num_cameras,
            augmented_camera_intrinsics=self.camera_intrinsic.repeat(self.num_cameras, 1, 1),
            image_augmentation_matrices=torch.eye(4, dtype=torch.float32).repeat(
                self.num_cameras, 1, 1
            ),
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def assert_augmentation_matrices_match_intrinsics(self, camera_image_data: BaseImages) -> None:
        """Assert the composed augmentation affines map the raw intrinsics onto the augmented ones.

        Args:
            camera_image_data: Camera image data returned by a transform.
        """
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                camera_image_data.image_augmentation_pixel_affines()
                @ camera_image_data.camera_intrinsics,
                atol=1e-5,
            )
        )

    def assert_projection_is_consistent(self, camera_image_data: BaseImages) -> None:
        """Assert `lidar2images` projects like the augmented intrinsics and `lidar2cams`.

        Args:
            camera_image_data: Camera image data returned by a transform.
        """
        homogeneous_intrinsics = torch.eye(4, dtype=torch.float32).repeat(self.num_cameras, 1, 1)
        homogeneous_intrinsics[:, :3, :3] = camera_image_data.augmented_camera_intrinsics
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images,
                homogeneous_intrinsics @ camera_image_data.lidar2cams,
                atol=1e-5,
            )
        )


class TestCropAndScale(CameraImageDataTestCase):
    """Unit tests for the CropAndScale transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        self.assertEqual(CropAndScale().crop_ratio, 0.8)
        self.assertEqual(CropAndScale(probability=0.9, crop_ratio=0.7).crop_ratio, 0.7)

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            CropAndScale(probability=1.0)(sample)

    def test_zero_probability_is_a_no_op(self) -> None:
        """Test that a zero probability leaves the sample untouched."""
        sample = self.build_multi_task_gt_sample()

        output = CropAndScale(probability=0.0)(sample)

        self.assertIs(output.camera_image_data, sample.camera_image_data)

    def test_image_size_preserved_and_intrinsics_updated(self) -> None:
        """Test that the crop is scaled back to the source size and zooms the intrinsics."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = CropAndScale(probability=1.0, crop_ratio=0.8)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.images.shape, sample.camera_image_data.images.shape)
        # Scaling a crop back to the source size can only magnify the focal lengths.
        self.assertTrue(torch.all(camera_image_data.augmented_camera_intrinsics[:, 0, 0] >= 100.0))
        self.assertTrue(torch.all(camera_image_data.augmented_camera_intrinsics[:, 1, 1] >= 100.0))
        # The raw intrinsics stay untouched, only the augmented ones follow the transform.
        self.assertTrue(
            torch.equal(
                camera_image_data.camera_intrinsics, sample.camera_image_data.camera_intrinsics
            )
        )
        self.assert_augmentation_matrices_match_intrinsics(camera_image_data)
        self.assert_projection_is_consistent(camera_image_data)

    def test_cameras_are_cropped_independently(self) -> None:
        """Test that every camera of a sample is given its own crop box."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()

        output = CropAndScale(probability=1.0, crop_ratio=0.5)(sample)

        assert output.camera_image_data is not None
        intrinsics = output.camera_image_data.augmented_camera_intrinsics
        self.assertFalse(torch.allclose(intrinsics[0], intrinsics[1]))


class TestResizeMultiviewImages(CameraImageDataTestCase):
    """Unit tests for the ResizeMultiviewImages transform."""

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            ResizeMultiviewImages(target_size=[16, 24])(sample)

    def test_images_and_intrinsics_are_scaled(self) -> None:
        """Test that halving the image size halves the focal lengths and the center."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeMultiviewImages(target_size=[self.image_height // 2, self.image_width // 2])(
            sample
        )

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(
            camera_image_data.images.shape,
            (self.num_cameras, 3, self.image_height // 2, self.image_width // 2),
        )
        expected_intrinsic = torch.tensor(
            [[50.0, 0.0, 12.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                expected_intrinsic.repeat(self.num_cameras, 1, 1),
            )
        )
        self.assert_augmentation_matrices_match_intrinsics(camera_image_data)
        self.assert_projection_is_consistent(camera_image_data)

    def test_anisotropic_resize(self) -> None:
        """Test that the width and the height are scaled independently."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeMultiviewImages(target_size=[self.image_height, self.image_width * 2])(
            sample
        )

        assert output.camera_image_data is not None
        intrinsics = output.camera_image_data.augmented_camera_intrinsics
        self.assertTrue(torch.allclose(intrinsics[:, 0, 0], torch.full((2,), 200.0)))
        self.assertTrue(torch.allclose(intrinsics[:, 1, 1], torch.full((2,), 100.0)))

    def test_projected_point_follows_the_resize(self) -> None:
        """Test that a lidar point projects onto the pixel the resize maps it to."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        # (4, ), a point five metres in front of the lidar.
        point = torch.tensor([1.0, 0.5, 5.0, 1.0], dtype=torch.float32)
        source_projection = sample.camera_image_data.lidar2images[0] @ point
        source_pixel = source_projection[:2] / source_projection[2]

        output = ResizeMultiviewImages(target_size=[self.image_height // 2, self.image_width // 2])(
            sample
        )

        assert output.camera_image_data is not None
        projection = output.camera_image_data.lidar2images[0] @ point
        self.assertTrue(
            torch.allclose(projection[:2] / projection[2], source_pixel / 2.0, atol=1e-5)
        )


class TestPadMultiViewImage(CameraImageDataTestCase):
    """Unit tests for the PadMultiViewImage transform."""

    def test_requires_exactly_one_size_argument(self) -> None:
        """Test that giving none or both size arguments raises ValueError."""
        with self.assertRaises(ValueError):
            PadMultiViewImage()
        with self.assertRaises(ValueError):
            PadMultiViewImage(size=[64, 64], size_divisor=32)

    def test_pads_to_size_divisor(self) -> None:
        """Test that the images are rounded up to a multiple of the size divisor."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = PadMultiViewImage(size_divisor=32, pad_value=0.0)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.images.shape, (self.num_cameras, 3, 32, 64))
        # The padding is appended at the bottom and the right, the source pixels stay put.
        self.assertTrue(
            torch.equal(
                camera_image_data.images[:, :, : self.image_height, : self.image_width],
                sample.camera_image_data.images,
            )
        )
        self.assertTrue(torch.all(camera_image_data.images[:, :, :, self.image_width :] == 0.0))

    def test_pads_to_fixed_size_and_keeps_intrinsics(self) -> None:
        """Test that padding to a fixed size leaves the intrinsics untouched."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = PadMultiViewImage(size=[64, 64], pad_value=1.0)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.images.shape, (self.num_cameras, 3, 64, 64))
        self.assertTrue(torch.all(camera_image_data.images[:, :, self.image_height :, :] == 1.0))
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                sample.camera_image_data.augmented_camera_intrinsics,
            )
        )
        self.assert_augmentation_matrices_match_intrinsics(camera_image_data)
        self.assert_projection_is_consistent(camera_image_data)

    def test_smaller_target_size_raises(self) -> None:
        """Test that padding to a smaller size than the images raises ValueError."""
        sample = self.build_multi_task_gt_sample()

        with self.assertRaises(ValueError):
            PadMultiViewImage(size=[16, 16])(sample)


class TestResizeCropFlipRotImage(CameraImageDataTestCase):
    """Unit tests for the ResizeCropFlipRotImage transform."""

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            ResizeCropFlipRotImage(
                target_size=[16, 24],
                resize_range=[0.5, 0.5],
                bottom_crop_ratio_range=[0.0, 0.0],
                training=False,
            )(sample)

    def test_target_size_and_intrinsics_in_validation(self) -> None:
        """Test that a plain half resize is deterministic and only scales the intrinsics."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height // 2, self.image_width // 2],
            resize_range=[0.5, 0.5],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(
            camera_image_data.images.shape,
            (self.num_cameras, 3, self.image_height // 2, self.image_width // 2),
        )
        expected_intrinsic = torch.tensor(
            [[50.0, 0.0, 12.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                expected_intrinsic.repeat(self.num_cameras, 1, 1),
                atol=1e-5,
            )
        )
        self.assert_augmentation_matrices_match_intrinsics(camera_image_data)
        self.assert_projection_is_consistent(camera_image_data)

    def test_scalar_resize_range_covers_the_target_size(self) -> None:
        """Test that a scalar resize range is read as a tolerance around a covering resize."""
        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height // 2, self.image_width // 4],
            resize_range=0.0,
            bottom_crop_ratio_range=[0.0, 0.0],
            training=True,
        )

        parameters = transform.sample_augmentation(self.image_height, self.image_width)

        # The target is narrower than the source: the resize covers the target height and
        # the excess width is cropped away instead of being zero-filled.
        self.assertEqual(parameters.resized_height, self.image_height // 2)
        self.assertEqual(parameters.resized_width, self.image_width // 2)
        self.assertEqual(parameters.crop_top, 0)
        self.assertGreaterEqual(parameters.crop_left, 0)
        self.assertLessEqual(parameters.crop_left, self.image_width // 2 - self.image_width // 4)

    def test_crop_is_placed_at_random_along_the_width_in_training(self) -> None:
        """Test that the crop box slides along the width in training and is centered otherwise."""
        np.random.seed(0)
        target_width = self.image_width // 2
        training_transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, target_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=True,
        )
        validation_transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, target_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        crop_lefts = {
            training_transform.sample_augmentation(self.image_height, self.image_width).crop_left
            for _ in range(20)
        }
        validation_parameters = validation_transform.sample_augmentation(
            self.image_height, self.image_width
        )

        self.assertGreater(len(crop_lefts), 1)
        self.assertTrue(
            all(0 <= crop_left <= self.image_width - target_width for crop_left in crop_lefts)
        )
        self.assertEqual(validation_parameters.crop_left, (self.image_width - target_width) // 2)

    def test_bottom_crop_ratio_drops_the_bottom_of_the_image(self) -> None:
        """Test that the bottom crop ratio lifts the crop box and zero-fills above the image."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        image = sample.camera_image_data.images[0]
        half_height = self.image_height // 2

        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.5, 0.5],
            training=False,
        )
        parameters = transform.sample_augmentation(self.image_height, self.image_width)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        # The bottom half of the image is dropped, the top half lands at the bottom of the
        # crop box and the rows above the image are zero-filled.
        self.assertEqual(parameters.crop_top, -half_height)
        self.assertEqual(parameters.crop_left, 0)
        self.assertTrue(
            torch.equal(augmented_image[:, :half_height], torch.zeros_like(image[:, :half_height]))
        )
        self.assertTrue(torch.allclose(augmented_image[:, half_height:], image[:, :half_height]))
        shifted_pixel = image_transform @ torch.tensor([12.0, 8.0, 1.0])
        self.assertAlmostEqual(float(shifted_pixel[0]), 12.0)
        self.assertAlmostEqual(float(shifted_pixel[1]), 8.0 + half_height)

    def test_horizontal_flip_matches_the_intrinsics(self) -> None:
        """Test that a flipped image is described by the flipped augmented intrinsics."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        image = sample.camera_image_data.images[0]

        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=True,
            random_horizontal_flip=True,
        )
        parameters = transform.sample_augmentation(self.image_height, self.image_width)
        parameters = parameters._replace(horizontal_flip=True)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        self.assertTrue(torch.allclose(augmented_image, torch.flip(image, dims=[-1])))
        # The principal point is mirrored about the center of the image width.
        principal_point = image_transform @ torch.tensor([24.0, 16.0, 1.0])
        self.assertAlmostEqual(float(principal_point[0]), self.image_width - 1.0 - 24.0)
        self.assertAlmostEqual(float(principal_point[1]), 16.0)

    def test_rotation_matches_the_intrinsics(self) -> None:
        """Test that the image transform predicts where a rotation moves a pixel."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        # A single lit pixel whose location can be recovered from the rotated image.
        image = torch.zeros_like(sample.camera_image_data.images[0])
        image[:, 8, 12] = 1.0

        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )
        parameters = transform.sample_augmentation(self.image_height, self.image_width)
        parameters = parameters._replace(rotation=90.0)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        rotated_pixel = image_transform @ torch.tensor([12.0, 8.0, 1.0])
        lit_index = int(torch.argmax(augmented_image[0].flatten()))
        lit_y, lit_x = divmod(lit_index, self.image_width)
        self.assertAlmostEqual(float(rotated_pixel[0]), lit_x, places=4)
        self.assertAlmostEqual(float(rotated_pixel[1]), lit_y, places=4)

    def test_cameras_are_augmented_independently(self) -> None:
        """Test that every camera of a sample is given its own augmentation."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[0.5, 1.5],
            bottom_crop_ratio_range=[0.0, 0.3],
            training=True,
        )(sample)

        assert output.camera_image_data is not None
        intrinsics = output.camera_image_data.augmented_camera_intrinsics
        self.assertFalse(torch.allclose(intrinsics[0], intrinsics[1]))
        self.assert_projection_is_consistent(output.camera_image_data)

    @staticmethod
    def intensity_centroid(
        image: Float32[Tensor, "num_channels height width"],
    ) -> Tuple[float, float]:
        """Locate the intensity-weighted centroid of an image in continuous pixel coordinates.

        Pixel ``(column, row)`` covers ``[column, column + 1) x [row, row + 1)``, the
        convention the resize follows, so its center sits at ``(column + 0.5, row + 0.5)``.

        Args:
            image: Image whose first channel holds the intensities.

        Returns:
            The centroid as ``(x, y)``.
        """
        weights = image[0]
        rows = torch.arange(weights.shape[0], dtype=torch.float32) + 0.5
        columns = torch.arange(weights.shape[1], dtype=torch.float32) + 0.5
        total = float(weights.sum())
        return (
            float((weights.sum(dim=0) * columns).sum() / total),
            float((weights.sum(dim=1) * rows).sum() / total),
        )

    def test_image_transform_matches_pixels_for_a_mismatched_aspect_ratio(self) -> None:
        """Test that the image transform describes the pixel mapping across aspect ratios.

        Regression test: with a source aspect ratio different from the target one, a hidden
        second resize used to stretch the pixels without entering the transform,
        desynchronizing the augmented intrinsics from the image.
        """
        source_height, source_width = 930, 1440
        image = torch.zeros((3, source_height, source_width), dtype=torch.float32)
        # A block whose center sits at (720, 465) in continuous pixel coordinates.
        image[:, 450:480, 705:735] = 1.0
        transform = ResizeCropFlipRotImage(
            target_size=[240, 320],
            resize_range=0.02,
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        parameters = transform.sample_augmentation(source_height, source_width)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        self.assertEqual(augmented_image.shape, (3, 240, 320))
        # The resize covers the target height, the excess width is cropped symmetrically.
        self.assertEqual((parameters.resized_height, parameters.resized_width), (240, 372))
        self.assertEqual((parameters.crop_top, parameters.crop_left), (0, 26))
        predicted = image_transform @ torch.tensor([720.0, 465.0, 1.0])
        centroid_x, centroid_y = self.intensity_centroid(augmented_image)
        self.assertAlmostEqual(centroid_x, float(predicted[0]), delta=0.05)
        self.assertAlmostEqual(centroid_y, float(predicted[1]), delta=0.05)

    def test_image_transform_holds_the_scale_of_the_whole_pixel_resize(self) -> None:
        """Test that the transform scale is the one of the rounded resized size.

        Regression test: ``source * resize`` is rounded to whole pixels before resizing, so
        the effective scale along each axis is ``resized / source`` rather than ``resize``.
        """
        source_height, source_width = 930, 1440
        image = torch.zeros((3, source_height, source_width), dtype=torch.float32)
        # A block whose center sits at (1300, 465), near the edge of the crop box.
        image[:, 450:480, 1285:1315] = 1.0
        transform = ResizeCropFlipRotImage(
            target_size=[240, 320],
            # 1440 * 0.2583 = 371.95, rounded to 372 whole pixels.
            resize_range=[0.2583, 0.2583],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        parameters = transform.sample_augmentation(source_height, source_width)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        self.assertEqual(parameters.resized_width, 372)
        self.assertAlmostEqual(float(image_transform[0, 0]) * source_width, 372.0, places=3)
        self.assertAlmostEqual(float(image_transform[1, 1]) * source_height, 240.0, places=3)
        predicted = image_transform @ torch.tensor([1300.0, 465.0, 1.0])
        centroid_x, centroid_y = self.intensity_centroid(augmented_image)
        self.assertAlmostEqual(centroid_x, float(predicted[0]), delta=0.05)
        self.assertAlmostEqual(centroid_y, float(predicted[1]), delta=0.05)


class TestImageAugmentationMatrices(CameraImageDataTestCase):
    """Unit tests for the composed image augmentation matrices."""

    # (4, ), a homogeneous lidar point five metres in front of the lidar, projecting in
    # front of every camera under test.
    LIDAR_POINT = torch.tensor([1.0, 0.5, 5.0, 1.0], dtype=torch.float32)

    def project_lidar_point(
        self, camera_image_data: BaseImages, camera_index: int
    ) -> Float32[Tensor, "3"]:
        """Project `LIDAR_POINT` onto the pixel plane of one camera.

        Args:
            camera_image_data: Camera image data holding the projection matrices.
            camera_index: Index of the camera the point is projected onto.

        Returns:
            The homogeneous pixel ``(u, v, 1)`` the point projects onto.
        """
        projection = camera_image_data.lidar2images[camera_index] @ self.LIDAR_POINT
        return projection[:3] / projection[2]

    def assert_inverse_recovers_the_source_pixel(
        self,
        source_camera_image_data: BaseImages,
        augmented_camera_image_data: BaseImages,
        atol: float = 1e-4,
    ) -> None:
        """Assert un-augmenting the augmented projection lands on the source pixel.

        The augmentation only remaps pixels, so the pixel a point projects onto in the
        augmented image, mapped back through the inverse of the composed affine, has to be
        the pixel the very same point projects onto in the source image.

        Args:
            source_camera_image_data: Camera image data before the transforms.
            augmented_camera_image_data: Camera image data returned by the transforms.
            atol: Absolute pixel tolerance of the comparison.
        """
        for camera_index in range(self.num_cameras):
            source_pixel = self.project_lidar_point(source_camera_image_data, camera_index)
            augmented_pixel = self.project_lidar_point(augmented_camera_image_data, camera_index)
            inverse_augmentation = torch.linalg.inv(
                augmented_camera_image_data.image_augmentation_pixel_affines()[camera_index]
            )
            recovered_pixel = inverse_augmentation @ augmented_pixel
            recovered_pixel = recovered_pixel / recovered_pixel[2]
            self.assertTrue(
                torch.allclose(recovered_pixel, source_pixel, atol=atol),
                f"camera {camera_index}: {recovered_pixel} != {source_pixel}",
            )

    def test_inverse_recovers_the_source_pixel_after_a_resize(self) -> None:
        """Test that a plain resize is undone by the inverse of the stored affine."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = ResizeMultiviewImages(target_size=[self.image_height, self.image_width * 2])(
            sample
        )

        assert output.camera_image_data is not None
        self.assert_inverse_recovers_the_source_pixel(
            sample.camera_image_data, output.camera_image_data
        )

    def test_inverse_recovers_the_source_pixel_after_a_crop_and_scale(self) -> None:
        """Test that a per-camera crop is undone by the inverse of the stored affine."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = CropAndScale(probability=1.0, crop_ratio=0.6)(sample)

        assert output.camera_image_data is not None
        self.assert_inverse_recovers_the_source_pixel(
            sample.camera_image_data, output.camera_image_data
        )

    def test_inverse_recovers_the_source_pixel_after_a_flip_and_rotation(self) -> None:
        """Test that a sampled resize, crop, flip, and rotation is undone by the inverse."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[0.8, 1.2],
            bottom_crop_ratio_range=[0.0, 0.2],
            training=True,
            random_horizontal_flip=True,
            rotation_range=[-5.0, 5.0],
        )(sample)

        assert output.camera_image_data is not None
        self.assert_inverse_recovers_the_source_pixel(
            sample.camera_image_data, output.camera_image_data
        )

    def test_inverse_recovers_the_source_pixel_after_a_chained_pipeline(self) -> None:
        """Test that the inverse undoes a whole chain of image-space transforms at once."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = ResizeMultiviewImages(target_size=[self.image_height, self.image_width])(
            PadMultiViewImage(size=[80, 128])(
                CropAndScale(probability=1.0, crop_ratio=0.7)(
                    ResizeMultiviewImages(
                        target_size=[self.image_height * 2, self.image_width * 2]
                    )(sample)
                )
            )
        )

        assert output.camera_image_data is not None
        self.assert_inverse_recovers_the_source_pixel(
            sample.camera_image_data, output.camera_image_data
        )

    def test_identity_before_any_transform(self) -> None:
        """Test that a freshly built sample carries the identity for every camera."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        self.assertTrue(
            torch.equal(
                sample.camera_image_data.image_augmentation_matrices,
                torch.eye(4, dtype=torch.float32).repeat(self.num_cameras, 1, 1),
            )
        )

    def test_holds_the_affine_of_a_single_resize(self) -> None:
        """Test that a half resize is stored as the matching scaling."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeMultiviewImages(target_size=[self.image_height // 2, self.image_width // 2])(
            sample
        )

        assert output.camera_image_data is not None
        expected = torch.tensor(
            [
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(
                output.camera_image_data.image_augmentation_matrices,
                expected.repeat(self.num_cameras, 1, 1),
                atol=1e-5,
            )
        )

    def test_stores_the_pixel_translation_in_the_homogeneous_layout(self) -> None:
        """Test that a crop lands its offset in the last column and leaves the depth axis alone."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.2, 0.2],
            training=True,
            random_horizontal_flip=False,
        )(sample)

        assert output.camera_image_data is not None
        matrices = output.camera_image_data.image_augmentation_matrices
        pixel_affines = output.camera_image_data.image_augmentation_pixel_affines()
        # The 2D part of the affine sits in the top-left block and the last column.
        self.assertTrue(torch.allclose(matrices[:, :2, :2], pixel_affines[:, :2, :2]))
        self.assertTrue(torch.allclose(matrices[:, :2, 3], pixel_affines[:, :2, 2]))
        self.assertTrue(torch.any(matrices[:, :2, 3] != 0.0))
        # The depth axis and the homogeneous row are the identity, so slicing ``[:3, 3]``
        # yields a zero depth offset and ``[:3, :3]`` a rotation padded with a unit depth.
        self.assertTrue(
            torch.equal(
                matrices[:, 2], torch.tensor([0.0, 0.0, 1.0, 0.0]).repeat(self.num_cameras, 1)
            )
        )
        self.assertTrue(
            torch.equal(
                matrices[:, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(self.num_cameras, 1)
            )
        )
        self.assertTrue(torch.equal(matrices[:, :2, 2], torch.zeros(self.num_cameras, 2)))
        # Lifting the 3x3 affines back reproduces the stored matrices.
        self.assertTrue(
            torch.allclose(
                BaseImages.homogeneous_image_augmentation_matrices(pixel_affines), matrices
            )
        )

    def test_composes_across_chained_transforms(self) -> None:
        """Test that chained transforms accumulate into a single affine."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()

        resized = ResizeMultiviewImages(target_size=[self.image_height, self.image_width * 2])(
            sample
        )
        assert resized.camera_image_data is not None
        cropped = CropAndScale(probability=1.0, crop_ratio=0.8)(resized)

        assert cropped.camera_image_data is not None
        # The affine of the second transform alone, recovered from the intrinsics it composed.
        second_transform = cropped.camera_image_data.augmented_camera_intrinsics @ torch.linalg.inv(
            resized.camera_image_data.augmented_camera_intrinsics
        )
        self.assertTrue(
            torch.allclose(
                cropped.camera_image_data.image_augmentation_pixel_affines(),
                second_transform @ resized.camera_image_data.image_augmentation_pixel_affines(),
                atol=1e-4,
            )
        )
        self.assert_augmentation_matrices_match_intrinsics(cropped.camera_image_data)

    def test_maps_a_source_pixel_onto_the_augmented_pixel(self) -> None:
        """Test that the stored affine moves a pixel the way the augmentation moved it."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        # (4, ), a point five metres in front of the lidar.
        point = torch.tensor([1.0, 0.5, 5.0, 1.0], dtype=torch.float32)
        source_projection = sample.camera_image_data.lidar2images[0] @ point
        source_pixel = source_projection[:3] / source_projection[2]

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[0.8, 1.2],
            bottom_crop_ratio_range=[0.0, 0.2],
            training=True,
            random_horizontal_flip=True,
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        augmented_projection = camera_image_data.lidar2images[0] @ point
        augmented_pixel = augmented_projection[:2] / augmented_projection[2]
        mapped_pixel = camera_image_data.image_augmentation_pixel_affines()[0] @ source_pixel
        self.assertTrue(
            torch.allclose(mapped_pixel[:2] / mapped_pixel[2], augmented_pixel, atol=1e-4)
        )

    def test_lidar_space_augmentation_leaves_the_matrices_untouched(self) -> None:
        """Test that moving the scene rather than the pixels keeps the affine unchanged."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        camera_image_data = sample.camera_image_data.update_lidar_transformation_matrices(
            torch.eye(4, dtype=torch.float32)
        )

        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                torch.eye(4, dtype=torch.float32).repeat(self.num_cameras, 1, 1),
            )
        )


if __name__ == "__main__":
    unittest.main()
