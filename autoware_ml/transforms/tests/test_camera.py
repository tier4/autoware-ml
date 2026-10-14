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

"""Unit tests for the camera transforms running on ``ModelGTSample``."""

from __future__ import annotations

from collections.abc import Sequence
import unittest

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.camera.distortion import UndistortImage
from autoware_ml.transforms.camera.resize import ResizeCropFlipRotImage
from autoware_ml.transforms.image.image import PhotometricDistortion
from autoware_ml.transforms.point_cloud.geometry import GlobalRotScaleTrans
from autoware_ml.types.geometry import PointFeatureName


def build_images(
    height: int = 8,
    width: int = 12,
    num_cameras: int = 2,
    distortion_coefficients: Sequence[float] = (),
) -> BaseImages:
    """Build images of the given size with a plain pinhole calibration."""
    camera_intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_cameras, 1, 1)
    camera_intrinsics[:, 0, 0] = float(width)
    camera_intrinsics[:, 1, 1] = float(height)
    camera_intrinsics[:, 0, 2] = width / 2.0
    camera_intrinsics[:, 1, 2] = height / 2.0
    lidar2cams = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
    return BaseImages(
        images=torch.rand((num_cameras, 3, height, width), dtype=torch.float32),
        timestamps=torch.zeros(num_cameras, dtype=torch.float64),
        camera_intrinsics=camera_intrinsics,
        camera_names=[f"CAM_{index}" for index in range(num_cameras)],
        lidar2images=torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1),
        lidar2cams=lidar2cams,
        distortion_models=["plumb_bob" if distortion_coefficients else ""] * num_cameras,
        distortion_coefficients=[torch.tensor(distortion_coefficients, dtype=torch.float32)]
        * num_cameras,
        augmented_camera_intrinsics=camera_intrinsics.clone(),
        image_augmentation_matrices=BaseImages.identity_image_augmentation_matrices(
            camera_intrinsics
        ),
    )


def build_sample(camera_image_data: BaseImages, with_points: bool = False) -> ModelGTSample:
    """Build a sample holding images and optionally a one point cloud."""
    point_cloud_data = (
        LiDARPoints(
            points=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            point_feature_names=[PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z],
            timestamp=0.0,
        )
        if with_points
        else None
    )
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=point_cloud_data,
        camera_image_data=camera_image_data,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )


class TestResizeCropFlipRotImage(unittest.TestCase):
    """Resize, crop and flip of the images and of their calibration."""

    def test_resizes_to_the_final_dimension(self) -> None:
        transform = ResizeCropFlipRotImage(
            target_size=[4, 6],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        sample = transform(build_sample(build_images()))

        self.assertEqual(sample.camera_image_data.images.shape[-2:], (4, 6))

    def test_composes_the_pixel_affine_into_the_intrinsics(self) -> None:
        # Halving the image halves the focal length and the principal point
        transform = ResizeCropFlipRotImage(
            target_size=[4, 6],
            resize_range=[0.5, 0.5],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        sample = transform(build_sample(build_images()))

        augmented = sample.camera_image_data.augmented_camera_intrinsics
        self.assertAlmostEqual(float(augmented[0, 0, 0]), 6.0, places=5)
        self.assertAlmostEqual(float(augmented[0, 1, 1]), 4.0, places=5)

    def test_composes_onto_the_intrinsics_of_earlier_image_transforms(self) -> None:
        # An earlier transform already halved the image, a second halving quarters it
        images = build_images()
        images = images.model_copy(
            update={"augmented_camera_intrinsics": images.augmented_camera_intrinsics * 0.5}
        )
        transform = ResizeCropFlipRotImage(
            target_size=[2, 3],
            resize_range=[0.5, 0.5],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        sample = transform(build_sample(images))

        augmented = sample.camera_image_data.augmented_camera_intrinsics
        self.assertAlmostEqual(float(augmented[0, 0, 0]), 3.0, places=5)

    def test_records_the_augmentation_matrix(self) -> None:
        transform = ResizeCropFlipRotImage(
            target_size=[4, 6],
            resize_range=[0.5, 0.5],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        sample = transform(build_sample(build_images()))

        affines = sample.camera_image_data.image_augmentation_pixel_affines()
        self.assertAlmostEqual(float(affines[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(affines[0, 1, 1]), 0.5, places=5)

    def test_keeps_lidar2image_consistent_with_the_intrinsics(self) -> None:
        transform = ResizeCropFlipRotImage(
            target_size=[4, 6],
            resize_range=[0.5, 0.5],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )

        sample = transform(build_sample(build_images()))

        images = sample.camera_image_data
        expected = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
        expected[:, :3, :3] = images.augmented_camera_intrinsics
        torch.testing.assert_close(images.lidar2images, expected @ images.lidar2cams)


class TestUndistortImage(unittest.TestCase):
    """Undistortion of the images and of their calibration."""

    def test_leaves_undistorted_images_untouched(self) -> None:
        images = build_images()
        transform = UndistortImage()

        sample = transform(build_sample(images))

        torch.testing.assert_close(sample.camera_image_data.images, images.images)

    def test_clears_the_distortion_it_removed(self) -> None:
        transform = UndistortImage()

        sample = transform(build_sample(build_images(distortion_coefficients=(0.1, 0.0, 0.0, 0.0))))

        images = sample.camera_image_data
        self.assertEqual([len(c) for c in images.distortion_coefficients], [0, 0])
        self.assertEqual(list(images.distortion_models), ["", ""])


class TestPhotometricDistortion(unittest.TestCase):
    """Color distortion of the images."""

    def test_identity_parameters_keep_the_images(self) -> None:
        images = build_images()
        transform = PhotometricDistortion(probability=1.0)

        sample = transform(build_sample(images))

        torch.testing.assert_close(
            sample.camera_image_data.images, images.images, atol=1e-5, rtol=0.0
        )

    def test_brightness_changes_the_images(self) -> None:
        images = build_images()
        transform = PhotometricDistortion(probability=1.0, brightness=0.5)

        sample = transform(build_sample(images))

        self.assertEqual(sample.camera_image_data.images.shape, images.images.shape)
        self.assertFalse(torch.equal(sample.camera_image_data.images, images.images))


class TestCameraFollowsTheLidarAugmentation(unittest.TestCase):
    """The lidar space augmentation updates the camera calibration."""

    def test_lidar2cam_takes_the_inverse_of_the_augmentation(self) -> None:
        # A pure scaling by two moves the points, so the cameras take the halving
        transform = GlobalRotScaleTrans(
            yaw_rot_range=[0.0, 0.0], scale_ratio_range=[2.0, 2.0], translation_std=None
        )

        sample = transform(build_sample(build_images(), with_points=True))

        lidar2cams = sample.camera_image_data.lidar2cams
        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, :3] = torch.eye(3, dtype=torch.float32) * 0.5
        torch.testing.assert_close(lidar2cams[0], expected, atol=1e-6, rtol=0.0)

    def test_sample_without_camera_stays_without_camera(self) -> None:
        transform = GlobalRotScaleTrans(
            yaw_rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=None
        )

        sample = transform(build_sample(None, with_points=True))

        self.assertIsNone(sample.camera_image_data)
