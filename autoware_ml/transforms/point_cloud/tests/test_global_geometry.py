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

"""Unit tests for the camera-lidar geometric augmentations."""

import unittest

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.point_cloud.geometry import GlobalBEVRandomFlip, GlobalRotScaleTrans
from autoware_ml.types.geometry import (
    Box3DCenterCoordinateType,
    Box3DFieldIndex,
    PointFeatureName,
    TransformationName,
)


class BaseCameraLidarGeometryTestCase(unittest.TestCase):
    """Shared sample builders and assertions for the camera-lidar geometric augmentations."""

    def build_bboxes_3d(self) -> LidarBBoxes3D:
        """Build a single 3D bounding box with a non-zero yaw and velocity.

        Returns:
            LidarBBoxes3D holding one bounding box.
        """
        return LidarBBoxes3D(
            bbox_params=torch.tensor(
                [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.3, 1.5, -0.5, 0.0]], dtype=torch.float32
            ),
            bbox_labels=torch.tensor([0], dtype=torch.int32),
            bbox_label_names=["car"],
            bbox_num_lidar_points=torch.tensor([10], dtype=torch.int32),
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        )

    def build_point_cloud_data(self) -> LiDARPoints:
        """Build a small point cloud with xyz and intensity features.

        Returns:
            LiDARPoints holding three points.
        """
        return LiDARPoints(
            points=torch.tensor(
                [
                    [1.0, 2.0, 0.5, 0.1],
                    [-3.0, 0.5, -1.0, 0.2],
                    [4.0, -2.0, 1.5, 0.3],
                ],
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

    def build_camera_image_data(self, num_cameras: int = 2) -> BaseImages:
        """Build camera data with identity matrices.

        Args:
            num_cameras: Number of cameras.

        Returns:
            BaseImages with identity camera matrices, hence every update applied by a
            transform is the transformation itself.
        """
        return BaseImages(
            images=torch.ones((num_cameras, 3, 4, 4), dtype=torch.float32),
            timestamps=torch.zeros(num_cameras, dtype=torch.float64),
            camera_intrinsics=torch.eye(3).repeat(num_cameras, 1, 1),
            camera_names=[f"camera{index}" for index in range(num_cameras)],
            lidar2images=torch.eye(4).repeat(num_cameras, 1, 1),
            lidar2cams=torch.eye(4).repeat(num_cameras, 1, 1),
            distortion_models=["plumb_bob"] * num_cameras,
            distortion_coefficients=[torch.zeros(5) for _ in range(num_cameras)],
            augmented_camera_intrinsics=torch.eye(3).repeat(num_cameras, 1, 1),
            image_augmentation_matrices=torch.eye(4).repeat(num_cameras, 1, 1),
        )

    def build_multi_task_gt_sample(
        self,
        with_point_cloud_data: bool = True,
        with_camera_image_data: bool = True,
        with_bboxes_3d: bool = True,
    ) -> ModelGTSample:
        """Build a minimal sample holding the requested modalities and optional bboxes.

        Args:
            with_point_cloud_data: Whether the sample holds a point cloud.
            with_camera_image_data: Whether the sample holds camera image data.
            with_bboxes_3d: Whether the sample holds 3D bounding boxes.

        Returns:
            ModelGTSample with the requested modalities.
        """
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=self.build_point_cloud_data() if with_point_cloud_data else None,
            camera_image_data=self.build_camera_image_data() if with_camera_image_data else None,
            detection3d_gt_bboxes_3d=self.build_bboxes_3d() if with_bboxes_3d else None,
            segmentation3d_gt_sample=None,
        )

    def apply_inverse_transformation(
        self,
        bboxes_3d: BaseBBoxes3D,
        transformation_matrix: Float32[Tensor, "4 4"],
    ) -> None:
        """Apply the inverse of a saved augmentation to the bboxes in place.

        The 4x4 matrix is decomposed back into the rotation, scale, and translation the
        bounding box API expects, which are applied in the same order the transforms use.

        Args:
            bboxes_3d: The augmented bounding boxes to restore.
            transformation_matrix: The 4x4 augmentation saved by the transform.
        """
        inverse_transformation_matrix = torch.linalg.inv(transformation_matrix)
        # The upper-left block is rotation * scale, so any of its columns has the scale as norm.
        scale_factor = float(torch.linalg.norm(inverse_transformation_matrix[:3, 0]))
        rotation_matrix = inverse_transformation_matrix[:3, :3] / scale_factor

        # Rotation is expressed in the column vector convention, the bboxes expect a row one.
        bboxes_3d.rotate(rotation_matrix.T)
        bboxes_3d.scale(scale_factor)
        bboxes_3d.translate(inverse_transformation_matrix[:3, 3].reshape(1, 3))

    def apply_transformation_to_xyz(
        self,
        xyz: Float32[Tensor, "num_points 3"],
        transformation_matrix: Float32[Tensor, "4 4"],
    ) -> Float32[Tensor, "num_points 3"]:
        """Apply a 4x4 homogeneous transformation to xyz coordinates.

        Args:
            xyz: The coordinates to transform.
            transformation_matrix: The 4x4 transformation in the column vector convention.

        Returns:
            The transformed coordinates.
        """
        homogeneous_xyz = torch.cat([xyz, torch.ones((xyz.shape[0], 1))], dim=1)
        return (transformation_matrix @ homogeneous_xyz.T).T[:, :3]

    def assert_bbox_params_close(
        self,
        bbox_params: Float32[Tensor, "num_bboxes num_Box3DFieldIndex"],
        expected_bbox_params: Float32[Tensor, "num_bboxes num_Box3DFieldIndex"],
    ) -> None:
        """Assert that two sets of bbox parameters match, comparing yaw modulo a full turn.

        Args:
            bbox_params: The bbox parameters under test.
            expected_bbox_params: The bbox parameters they are expected to match.
        """
        # A flip mirrors the yaw twice, which restores the heading but can offset it by 2 pi.
        yaw_difference = (
            torch.remainder(
                bbox_params[:, Box3DFieldIndex.YAW]
                - expected_bbox_params[:, Box3DFieldIndex.YAW]
                + torch.pi,
                2 * torch.pi,
            )
            - torch.pi
        )
        self.assertTrue(torch.allclose(yaw_difference, torch.zeros_like(yaw_difference), atol=1e-5))

        remaining_fields = [
            index for index in range(bbox_params.shape[1]) if index != Box3DFieldIndex.YAW
        ]
        self.assertTrue(
            torch.allclose(
                bbox_params[:, remaining_fields],
                expected_bbox_params[:, remaining_fields],
                atol=1e-5,
            )
        )

    def assert_camera_matrices_follow(
        self,
        camera_image_data: BaseImages,
        original_lidar2cams: Float32[Tensor, "num_cameras 4 4"],
        transformation_matrix: Float32[Tensor, "4 4"],
    ) -> None:
        """Assert that the camera extrinsics were re-expressed in the augmented frame.

        Args:
            camera_image_data: The camera data after the transform.
            original_lidar2cams: The lidar-to-camera matrices before the transform.
            transformation_matrix: The 4x4 augmentation saved by the transform.
        """
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2cams,
                original_lidar2cams @ torch.linalg.inv(transformation_matrix),
                atol=1e-5,
            )
        )
        # lidar2image is recomputed from the (unchanged) intrinsics and the new lidar2cam.
        expected_camera_intrinsics = torch.eye(4).repeat(camera_image_data.images.shape[0], 1, 1)
        expected_camera_intrinsics[:, :3, :3] = camera_image_data.augmented_camera_intrinsics
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images,
                expected_camera_intrinsics @ camera_image_data.lidar2cams,
                atol=1e-5,
            )
        )


class TestGlobalRotScaleTrans(BaseCameraLidarGeometryTestCase):
    """Unit tests for the camera-lidar GlobalRotScaleTrans transform."""

    def setUp(self) -> None:
        """Set up the same fusion sample and transform parameters for all tests."""
        np.random.seed(0)
        self.sample = self.build_multi_task_gt_sample()
        assert self.sample.detection3d_gt_bboxes_3d is not None
        assert self.sample.point_cloud_data is not None
        assert self.sample.camera_image_data is not None
        self.original_bbox_params = self.sample.detection3d_gt_bboxes_3d.bbox_params.clone()
        self.original_points = self.sample.point_cloud_data.points.clone()
        self.original_lidar2cams = self.sample.camera_image_data.lidar2cams.clone()
        self.original_lidar2images = self.sample.camera_image_data.lidar2images.clone()
        self.yaw_rot_range = [-0.5, 0.5]
        self.scale_ratio_range = [0.9, 1.1]
        self.translation_std = [0.5, 0.5, 0.2]

    def build_transform(self) -> GlobalRotScaleTrans:
        """Build the transform with the randomized parameters of the test case."""
        return GlobalRotScaleTrans(
            yaw_rot_range=self.yaw_rot_range,
            scale_ratio_range=self.scale_ratio_range,
            translation_std=self.translation_std,
        )

    def test_saves_transformation_matrix(self) -> None:
        """Test that the sampled augmentation is saved as a 4x4 transformation matrix."""
        transform = self.build_transform()

        # Both draws start from the same seed, so the transform samples the expected values.
        np.random.seed(0)
        expected_lidar_transformation_sample, _ = transform.sample_rot_scale_trans()
        np.random.seed(0)
        output = transform(self.sample)

        assert output.lidar_transformation_sample is not None
        lidar_transformation_sample = output.lidar_transformation_sample
        self.assertTrue(
            torch.allclose(
                lidar_transformation_sample.transformation_matrix,
                expected_lidar_transformation_sample.transformation_matrix,
            )
        )
        self.assertEqual(
            lidar_transformation_sample.transformation_order,
            [
                TransformationName.ROTATION,
                TransformationName.SCALING,
                TransformationName.TRANSLATION,
            ],
        )

    def test_points_follow_the_saved_transformation(self) -> None:
        """Test that the points are transformed exactly by the saved 4x4 matrix."""
        output = self.build_transform()(self.sample)

        assert output.point_cloud_data is not None
        assert output.lidar_transformation_sample is not None
        expected_xyz = self.apply_transformation_to_xyz(
            self.original_points[:, :3], output.lidar_transformation_sample.transformation_matrix
        )
        self.assertTrue(
            torch.allclose(output.point_cloud_data.points[:, :3], expected_xyz, atol=1e-5)
        )
        # Non-spatial features are untouched.
        self.assertTrue(
            torch.allclose(output.point_cloud_data.points[:, 3:], self.original_points[:, 3:])
        )

    def test_camera_matrices_follow_the_augmentation(self) -> None:
        """Test that the camera extrinsics are re-expressed in the augmented frame."""
        output = self.build_transform()(self.sample)

        assert output.lidar_transformation_sample is not None
        assert output.camera_image_data is not None
        self.assert_camera_matrices_follow(
            output.camera_image_data,
            self.original_lidar2cams,
            output.lidar_transformation_sample.transformation_matrix,
        )

    def test_bboxes_transformed(self) -> None:
        """Test that the bboxes are rotated, scaled, and translated."""
        output = GlobalRotScaleTrans(yaw_rot_range=[0.3, 0.3], scale_ratio_range=[2.0, 2.0])(
            self.sample
        )

        assert output.detection3d_gt_bboxes_3d is not None
        bbox_params = output.detection3d_gt_bboxes_3d.bbox_params
        # Yaw is offset by the sampled rotation and the dimensions scaled by the sampled scale.
        self.assertTrue(
            torch.allclose(bbox_params[:, 6], self.original_bbox_params[:, 6] + 0.3, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(bbox_params[:, 3:6], self.original_bbox_params[:, 3:6] * 2.0, atol=1e-5)
        )

    def test_scale_applies_to_bbox_centers_dimensions_and_velocity(self) -> None:
        """Test that a pure scale shrinks the bbox center, dimensions and velocity alike, not yaw."""
        output = GlobalRotScaleTrans(yaw_rot_range=[0.0, 0.0], scale_ratio_range=[0.5, 0.5])(
            self.sample
        )

        assert output.detection3d_gt_bboxes_3d is not None
        bbox_params = output.detection3d_gt_bboxes_3d.bbox_params
        # Original box: center (1, 2, 0), dims (4, 2, 1), yaw 0.3, velocity (1.5, -0.5).
        self.assertTrue(
            torch.allclose(bbox_params[0, :3], torch.tensor([0.5, 1.0, 0.0]), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(bbox_params[0, 3:6], torch.tensor([2.0, 1.0, 0.5]), atol=1e-5)
        )
        self.assertAlmostEqual(float(bbox_params[0, 6]), 0.3, places=5)
        # Velocity is a spatial quantity per unit time, so it scales with space.
        self.assertTrue(torch.allclose(bbox_params[0, 7:9], torch.tensor([0.75, -0.25]), atol=1e-5))

    def test_bbox_centers_and_points_share_the_transformation(self) -> None:
        """Test that the bbox centers move exactly like the points would."""
        output = self.build_transform()(self.sample)

        assert output.detection3d_gt_bboxes_3d is not None
        assert output.lidar_transformation_sample is not None
        expected_centers = self.apply_transformation_to_xyz(
            self.original_bbox_params[:, :3],
            output.lidar_transformation_sample.transformation_matrix,
        )
        self.assertTrue(
            torch.allclose(
                output.detection3d_gt_bboxes_3d.bbox_params[:, :3], expected_centers, atol=1e-5
            )
        )

    def test_lidar_only_sample(self) -> None:
        """Test that a sample without cameras is transformed and the cameras stay None."""
        sample = self.build_multi_task_gt_sample(with_camera_image_data=False)
        output = self.build_transform()(sample)

        assert output.point_cloud_data is not None
        assert output.lidar_transformation_sample is not None
        self.assertIsNone(output.camera_image_data)
        expected_xyz = self.apply_transformation_to_xyz(
            self.original_points[:, :3], output.lidar_transformation_sample.transformation_matrix
        )
        self.assertTrue(
            torch.allclose(output.point_cloud_data.points[:, :3], expected_xyz, atol=1e-5)
        )

    def test_camera_only_sample(self) -> None:
        """Test that a sample without points updates the cameras and leaves the points None."""
        sample = self.build_multi_task_gt_sample(with_point_cloud_data=False)
        output = self.build_transform()(sample)

        assert output.camera_image_data is not None
        assert output.lidar_transformation_sample is not None
        self.assertIsNone(output.point_cloud_data)
        self.assert_camera_matrices_follow(
            output.camera_image_data,
            self.original_lidar2cams,
            output.lidar_transformation_sample.transformation_matrix,
        )

    def test_without_bboxes(self) -> None:
        """Test that a sample without bboxes is transformed."""
        sample = self.build_multi_task_gt_sample(with_bboxes_3d=False)
        output = self.build_transform()(sample)

        self.assertIsNone(output.detection3d_gt_bboxes_3d)
        self.assertIsNotNone(output.lidar_transformation_sample)

    def test_composes_with_previous_transformation(self) -> None:
        """Test that the augmentation is composed with an earlier one in the pipeline."""
        transform = GlobalRotScaleTrans(yaw_rot_range=[0.3, 0.3], scale_ratio_range=[2.0, 2.0])

        first_output = transform(self.sample)
        assert first_output.lidar_transformation_sample is not None
        first_transformation_matrix = (
            first_output.lidar_transformation_sample.transformation_matrix.clone()
        )
        second_output = transform(first_output)

        assert second_output.lidar_transformation_sample is not None
        self.assertTrue(
            torch.allclose(
                second_output.lidar_transformation_sample.transformation_matrix,
                first_transformation_matrix @ first_transformation_matrix,
                atol=1e-5,
            )
        )
        self.assertEqual(
            second_output.lidar_transformation_sample.transformation_order,
            [
                TransformationName.ROTATION,
                TransformationName.SCALING,
                TransformationName.TRANSLATION,
            ]
            * 2,
        )

    def test_inverse_transformation_recovers_original_bboxes(self) -> None:
        """Test that inversely applying the saved matrix restores the original bboxes."""
        output = self.build_transform()(self.sample)

        assert output.detection3d_gt_bboxes_3d is not None
        assert output.lidar_transformation_sample is not None
        bboxes_3d = output.detection3d_gt_bboxes_3d
        self.assertFalse(torch.allclose(bboxes_3d.bbox_params, self.original_bbox_params))

        self.apply_inverse_transformation(
            bboxes_3d, output.lidar_transformation_sample.transformation_matrix
        )

        self.assert_bbox_params_close(bboxes_3d.bbox_params, self.original_bbox_params)

    def test_inverse_transformation_recovers_original_points(self) -> None:
        """Test that inversely applying the saved matrix restores the original points."""
        output = self.build_transform()(self.sample)

        assert output.point_cloud_data is not None
        assert output.lidar_transformation_sample is not None
        points = output.point_cloud_data.points
        self.assertFalse(torch.allclose(points, self.original_points))

        restored_xyz = self.apply_transformation_to_xyz(
            points[:, :3],
            torch.linalg.inv(output.lidar_transformation_sample.transformation_matrix),
        )
        self.assertTrue(torch.allclose(restored_xyz, self.original_points[:, :3], atol=1e-5))

    def test_inverse_transformation_recovers_original_camera_matrices(self) -> None:
        """Test that inversely applying the saved matrix restores the camera matrices."""
        output = self.build_transform()(self.sample)

        assert output.camera_image_data is not None
        assert output.lidar_transformation_sample is not None
        camera_image_data = output.camera_image_data
        transformation_matrix = output.lidar_transformation_sample.transformation_matrix
        self.assertFalse(torch.allclose(camera_image_data.lidar2cams, self.original_lidar2cams))

        # The extrinsics were composed with the inverse of the augmentation, so composing
        # them with the augmentation itself brings them back.
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2cams @ transformation_matrix,
                self.original_lidar2cams,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images @ transformation_matrix,
                self.original_lidar2images,
                atol=1e-5,
            )
        )

    def test_missing_both_modalities(self) -> None:
        """Test that a sample with neither points nor cameras raises KeyError."""
        sample = self.build_multi_task_gt_sample(
            with_point_cloud_data=False, with_camera_image_data=False
        )
        transform = GlobalRotScaleTrans(yaw_rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0])

        with self.assertRaises(KeyError):
            transform(sample)


class TestGlobalBEVRandomFlip(BaseCameraLidarGeometryTestCase):
    """Unit tests for the camera-lidar GlobalBEVRandomFlip transform."""

    def setUp(self) -> None:
        """Set up the same fusion sample for all tests."""
        self.sample = self.build_multi_task_gt_sample()
        assert self.sample.detection3d_gt_bboxes_3d is not None
        assert self.sample.point_cloud_data is not None
        assert self.sample.camera_image_data is not None
        self.original_bbox_params = self.sample.detection3d_gt_bboxes_3d.bbox_params.clone()
        self.original_points = self.sample.point_cloud_data.points.clone()
        self.original_lidar2cams = self.sample.camera_image_data.lidar2cams.clone()
        self.original_lidar2images = self.sample.camera_image_data.lidar2images.clone()

    def test_flips_points_bboxes_and_camera_matrices(self) -> None:
        """Test that both BEV flips negate x and y for the points, bboxes and cameras."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(
            self.sample
        )

        expected_flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        assert output.lidar_transformation_sample is not None
        assert output.point_cloud_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        assert output.camera_image_data is not None
        lidar_transformation_sample = output.lidar_transformation_sample
        self.assertTrue(
            torch.allclose(lidar_transformation_sample.transformation_matrix, expected_flip)
        )
        self.assertEqual(
            lidar_transformation_sample.transformation_order,
            [TransformationName.HORIZONTAL_FLIP, TransformationName.VERTICAL_FLIP],
        )
        self.assertTrue(
            torch.allclose(
                output.point_cloud_data.points[:, :2], -self.original_points[:, :2], atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(output.point_cloud_data.points[:, 2:], self.original_points[:, 2:])
        )
        self.assertTrue(
            torch.allclose(
                output.detection3d_gt_bboxes_3d.bbox_params[:, :2],
                -self.original_bbox_params[:, :2],
                atol=1e-5,
            )
        )
        self.assert_camera_matrices_follow(
            output.camera_image_data, self.original_lidar2cams, expected_flip
        )

    def test_no_flip_keeps_identity(self) -> None:
        """Test that a zero flip ratio saves an identity transformation."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=0.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.lidar_transformation_sample is not None
        assert output.point_cloud_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        assert output.camera_image_data is not None
        self.assertTrue(
            torch.allclose(output.lidar_transformation_sample.transformation_matrix, torch.eye(4))
        )
        self.assertEqual(output.lidar_transformation_sample.transformation_order, [])
        self.assertTrue(torch.allclose(output.point_cloud_data.points, self.original_points))
        self.assertTrue(
            torch.allclose(output.detection3d_gt_bboxes_3d.bbox_params, self.original_bbox_params)
        )
        self.assertTrue(
            torch.allclose(output.camera_image_data.lidar2cams, self.original_lidar2cams)
        )

    def test_horizontal_flip_only_negates_lateral_axis(self) -> None:
        """Test that a horizontal flip mirrors the lateral (y) axis only."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.lidar_transformation_sample is not None
        assert output.point_cloud_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        self.assertTrue(
            torch.allclose(
                output.lidar_transformation_sample.transformation_matrix,
                torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0])),
            )
        )
        points = output.point_cloud_data.points
        self.assertTrue(torch.allclose(points[:, 0], self.original_points[:, 0]))
        self.assertTrue(torch.allclose(points[:, 1], -self.original_points[:, 1]))
        bbox_params = output.detection3d_gt_bboxes_3d.bbox_params
        self.assertTrue(torch.allclose(bbox_params[:, 0], self.original_bbox_params[:, 0]))
        self.assertTrue(torch.allclose(bbox_params[:, 1], -self.original_bbox_params[:, 1]))

    def test_lidar_only_sample(self) -> None:
        """Test that a sample without cameras is flipped and the cameras stay None."""
        sample = self.build_multi_task_gt_sample(with_camera_image_data=False)
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=0.0, vertical_flip_ratio=1.0)(sample)

        assert output.point_cloud_data is not None
        self.assertIsNone(output.camera_image_data)
        points = output.point_cloud_data.points
        self.assertTrue(torch.allclose(points[:, 0], -self.original_points[:, 0]))
        self.assertTrue(torch.allclose(points[:, 1], self.original_points[:, 1]))

    def test_camera_only_sample(self) -> None:
        """Test that a sample without points updates the cameras and leaves the points None."""
        sample = self.build_multi_task_gt_sample(with_point_cloud_data=False)
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(sample)

        assert output.camera_image_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        self.assertIsNone(output.point_cloud_data)
        expected_flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        self.assert_camera_matrices_follow(
            output.camera_image_data, self.original_lidar2cams, expected_flip
        )
        self.assertTrue(
            torch.allclose(
                output.detection3d_gt_bboxes_3d.bbox_params[:, :2],
                -self.original_bbox_params[:, :2],
                atol=1e-5,
            )
        )

    def test_composes_with_previous_transformation(self) -> None:
        """Test that the flip is composed with an earlier transformation in the pipeline."""
        transform = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)

        output = transform(transform(self.sample))

        assert output.lidar_transformation_sample is not None
        assert output.point_cloud_data is not None
        # Flipping twice along the same axis returns to the identity.
        self.assertTrue(
            torch.allclose(output.lidar_transformation_sample.transformation_matrix, torch.eye(4))
        )
        self.assertEqual(
            output.lidar_transformation_sample.transformation_order,
            [TransformationName.HORIZONTAL_FLIP, TransformationName.HORIZONTAL_FLIP],
        )
        self.assertTrue(torch.allclose(output.point_cloud_data.points, self.original_points))

    def test_inverse_transformation_recovers_original_bboxes(self) -> None:
        """Test that inversely applying the saved matrix restores the original bboxes."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(
            self.sample
        )

        assert output.detection3d_gt_bboxes_3d is not None
        assert output.lidar_transformation_sample is not None
        bboxes_3d = output.detection3d_gt_bboxes_3d
        self.assertFalse(torch.allclose(bboxes_3d.bbox_params, self.original_bbox_params))

        self.apply_inverse_transformation(
            bboxes_3d, output.lidar_transformation_sample.transformation_matrix
        )

        self.assert_bbox_params_close(bboxes_3d.bbox_params, self.original_bbox_params)

    def test_inverse_transformation_recovers_original_camera_matrices(self) -> None:
        """Test that inversely applying the saved matrix restores the camera matrices."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.camera_image_data is not None
        assert output.lidar_transformation_sample is not None
        camera_image_data = output.camera_image_data
        transformation_matrix = output.lidar_transformation_sample.transformation_matrix
        self.assertFalse(torch.allclose(camera_image_data.lidar2cams, self.original_lidar2cams))

        # The extrinsics were composed with the inverse of the flip, so composing them with
        # the flip itself brings them back.
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2cams @ transformation_matrix,
                self.original_lidar2cams,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images @ transformation_matrix,
                self.original_lidar2images,
                atol=1e-5,
            )
        )

    def test_missing_both_modalities(self) -> None:
        """Test that a sample with neither points nor cameras raises KeyError."""
        sample = self.build_multi_task_gt_sample(
            with_point_cloud_data=False, with_camera_image_data=False
        )

        with self.assertRaises(KeyError):
            GlobalBEVRandomFlip()(sample)


if __name__ == "__main__":
    unittest.main()
