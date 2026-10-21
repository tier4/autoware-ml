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

"""Unit tests for the LiDARDepthSparseTransform transform."""

import unittest

from jaxtyping import Float32
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.camera_lidar.depth import LiDARDepthSparseTransform
from autoware_ml.types.geometry import PointFeatureName


class TestLiDARDepthSparseTransform(unittest.TestCase):
    """Unit tests for the LiDARDepthSparseTransform transform."""

    def setUp(self) -> None:
        """Set up the image size and the pinhole camera shared by every test."""
        self.height = 8
        self.width = 12
        self.focal_length = 10.0
        self.center_x = self.width / 2  # 6.0
        self.center_y = self.height / 2  # 4.0

    def build_camera_intrinsics(self) -> Float32[Tensor, "3 3"]:
        """Build a pinhole intrinsic matrix centered on the image.

        Returns:
            The 3x3 intrinsic matrix.
        """
        return torch.tensor(
            [
                [self.focal_length, 0.0, self.center_x],
                [0.0, self.focal_length, self.center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def build_lidar2cams(self, camera_translations: Float32[Tensor, "num_cameras 3"]) -> Tensor:
        """Build lidar-to-camera extrinsics that only translate the points.

        The camera frame is aligned with the lidar frame, so the camera looks along +z of the
        lidar and a point at ``(x, y, z)`` lands at camera coordinates ``(x, y, z) + t``.

        Args:
            camera_translations: Translation applied to the points for each camera.

        Returns:
            The 4x4 lidar-to-camera matrices.
        """
        num_cameras = camera_translations.shape[0]
        lidar2cams = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
        lidar2cams[:, :3, 3] = camera_translations
        return lidar2cams

    def build_multi_task_gt_sample(
        self,
        points_xyz: Float32[Tensor, "num_points 3"],
        camera_translations: Float32[Tensor, "num_cameras 3"] | None = None,
    ) -> ModelGTSample:
        """Build a sample with the given points and cameras.

        Args:
            points_xyz: The lidar points, an intensity column is appended.
            camera_translations: Per-camera translations, a single camera at the lidar origin
                by default.

        Returns:
            ModelGTSample holding the points and cameras with consistent projections.
        """
        if camera_translations is None:
            camera_translations = torch.zeros((1, 3), dtype=torch.float32)
        num_cameras = camera_translations.shape[0]

        camera_intrinsics = self.build_camera_intrinsics().repeat(num_cameras, 1, 1)
        homogeneous_intrinsics = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
        homogeneous_intrinsics[:, :3, :3] = camera_intrinsics
        lidar2cams = self.build_lidar2cams(camera_translations)

        camera_image_data = BaseImages(
            images=torch.ones((num_cameras, 3, self.height, self.width), dtype=torch.float32),
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
        intensity = torch.full((points_xyz.shape[0], 1), 0.5, dtype=torch.float32)
        point_cloud_data = LiDARPoints(
            points=torch.cat([points_xyz, intensity], dim=1),
            point_feature_names=[
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=point_cloud_data,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def test_points_land_on_the_expected_pixels_with_their_depth(self) -> None:
        """Test that the projected points write their depth at the right pixel, zeros elsewhere."""
        # A point on the optical axis lands on the image center, a point offset by 1 m in x at
        # 5 m depth lands focal_length * 1 / 5 = 2 pixels to the right of it.
        points_xyz = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(points_xyz)

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        depth_images = output.camera_image_data.depth_maps
        assert depth_images is not None
        self.assertEqual(depth_images.shape, (1, 1, self.height, self.width))
        self.assertEqual(depth_images.dtype, torch.float32)
        expected_depth_images = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32)
        expected_depth_images[0, 0, int(self.center_y), int(self.center_x)] = 5.0
        expected_depth_images[0, 0, int(self.center_y), int(self.center_x) + 2] = 5.0
        self.assertTrue(torch.equal(depth_images, expected_depth_images))

    def test_points_behind_the_camera_are_dropped(self) -> None:
        """Test that points with a non-positive depth never write to the depth map."""
        points_xyz = torch.tensor([[0.0, 0.0, -5.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(points_xyz)

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        assert output.camera_image_data.depth_maps is not None
        self.assertEqual(float(output.camera_image_data.depth_maps.abs().sum()), 0.0)

    def test_points_outside_the_image_are_dropped(self) -> None:
        """Test that points projecting outside the image bounds never write to the depth map."""
        # At 1 m depth a 1 m offset moves the pixel by the focal length, i.e. off a 12x8 image.
        points_xyz = torch.tensor(
            [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]],
            dtype=torch.float32,
        )
        sample = self.build_multi_task_gt_sample(points_xyz)

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        assert output.camera_image_data.depth_maps is not None
        self.assertEqual(float(output.camera_image_data.depth_maps.abs().sum()), 0.0)

    def test_each_camera_gets_its_own_projection(self) -> None:
        """Test that the same point lands on different pixels of differently placed cameras."""
        # The second camera is shifted by 1 m along x, so the point sits 1 m further to the
        # right in its frame: 2 pixels to the right at 5 m depth.
        camera_translations = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        points_xyz = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(points_xyz, camera_translations)

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        depth_images = output.camera_image_data.depth_maps
        assert depth_images is not None
        self.assertEqual(depth_images.shape, (2, 1, self.height, self.width))
        expected_depth_images = torch.zeros((2, 1, self.height, self.width), dtype=torch.float32)
        expected_depth_images[0, 0, int(self.center_y), int(self.center_x)] = 5.0
        expected_depth_images[1, 0, int(self.center_y), int(self.center_x) + 2] = 5.0
        self.assertTrue(torch.equal(depth_images, expected_depth_images))

    def test_depth_is_the_distance_along_the_optical_axis(self) -> None:
        """Test that a camera shifted along z sees the point at its own depth."""
        camera_translations = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32)
        points_xyz = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(points_xyz, camera_translations)

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        assert output.camera_image_data.depth_maps is not None
        self.assertEqual(
            float(
                output.camera_image_data.depth_maps[0, 0, int(self.center_y), int(self.center_x)]
            ),
            3.0,
        )

    def test_empty_point_cloud_gives_zero_depth(self) -> None:
        """Test that a sample without points gets all-zero depth maps."""
        sample = self.build_multi_task_gt_sample(torch.zeros((0, 3), dtype=torch.float32))

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        depth_images = output.camera_image_data.depth_maps
        assert depth_images is not None
        self.assertTrue(torch.equal(depth_images, torch.zeros((1, 1, self.height, self.width))))

    def test_other_fields_are_preserved(self) -> None:
        """Test that only the depth images change, the input container is left untouched."""
        points_xyz = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(points_xyz)
        assert sample.camera_image_data is not None
        assert sample.point_cloud_data is not None

        output = LiDARDepthSparseTransform()(sample)

        assert output.camera_image_data is not None
        original = sample.camera_image_data
        with_depth = output.camera_image_data
        self.assertIsNone(original.depth_maps)
        self.assertIsNotNone(with_depth.depth_maps)
        self.assertTrue(torch.equal(with_depth.images, original.images))
        self.assertTrue(torch.equal(with_depth.lidar2images, original.lidar2images))
        self.assertTrue(torch.equal(with_depth.lidar2cams, original.lidar2cams))
        self.assertTrue(torch.equal(with_depth.camera_intrinsics, original.camera_intrinsics))
        self.assertEqual(with_depth.camera_names, original.camera_names)
        self.assertIs(output.point_cloud_data, sample.point_cloud_data)

    def test_image_augmentation_baked_into_the_projection_is_honoured(self) -> None:
        """Test that an image affine composed into ``lidar2image`` moves the hits accordingly."""
        transform = LiDARDepthSparseTransform()
        # On the optical axis and 1 m to the right at 5 m depth: pixels (y, x) = (4, 6) and (4, 8).
        points = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=torch.float32)
        # A 2D affine that flips horizontally and shifts the image down by one pixel, composed
        # into the intrinsics the way the image transforms keep ``lidar2images`` in sync.
        image_affine = torch.tensor(
            [[-1.0, 0.0, self.width - 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        lidar2image = torch.eye(4, dtype=torch.float32)
        lidar2image[:3, :3] = image_affine @ self.build_camera_intrinsics()

        depth_maps = transform.build_depth_maps(
            points, lidar2image.view(1, 4, 4), (self.height, self.width)
        )

        self.assertEqual(depth_maps.shape, (1, self.height, self.width))
        expected_depth_maps = torch.zeros((1, self.height, self.width), dtype=torch.float32)
        flipped_center_x = self.width - 1 - int(self.center_x)
        shifted_center_y = int(self.center_y) + 1
        expected_depth_maps[0, shifted_center_y, flipped_center_x] = 5.0
        expected_depth_maps[0, shifted_center_y, flipped_center_x - 2] = 5.0
        self.assertTrue(torch.equal(depth_maps, expected_depth_maps))

    def test_missing_keys(self) -> None:
        """Test that a missing point cloud or camera data raises KeyError."""
        sample = self.build_multi_task_gt_sample(torch.tensor([[0.0, 0.0, 5.0]]))

        with self.assertRaises(KeyError):
            LiDARDepthSparseTransform()(sample._replace(point_cloud_data=None))
        with self.assertRaises(KeyError):
            LiDARDepthSparseTransform()(sample._replace(camera_image_data=None))


if __name__ == "__main__":
    unittest.main()
