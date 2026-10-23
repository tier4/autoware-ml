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

"""Unit tests for the camera-lidar geometric augmentations.

The camera fixtures carry non-trivial intrinsics and extrinsics that differ per camera, and
the augmented intrinsics differ from the raw ones, so a transform mishandling any of the
matrices (wrong side of the multiplication, missing inverse, rebuilding ``lidar2images`` from
the raw intrinsics, or touching the intrinsics) fails. The expected matrices are composed
with numpy from the sampled yaw, scale and translation, independently of the transform.
"""

import math
import unittest

from jaxtyping import Float32
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch, FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.tests.pinhole_camera_test_case import PinholeCameraTestCase
from autoware_ml.transforms.camera_lidar.geometry import (
    GlobalBEVRandomFlip,
    GlobalRotScaleTrans,
    RandomRotateTargetAngle,
)
from autoware_ml.types.geometry import (
    Box3DCenterCoordinateType,
    Box3DFieldIndex,
    PointFeatureName,
    TransformationName,
)

# Maps the lidar axes (x forward, y left, z up) onto the optical axes (x right, y down,
# z forward) of a camera looking straight ahead.
LIDAR_TO_OPTICAL_AXES = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def rotation_about_x(angle: float) -> npt.NDArray[np.float64]:
    """Build the rotation about the x axis in column vector convention."""
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=np.float64)


def rotation_about_y(angle: float) -> npt.NDArray[np.float64]:
    """Build the rotation about the y axis in column vector convention."""
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float64)


def rotation_about_z(angle: float) -> npt.NDArray[np.float64]:
    """Build the rotation about the z axis in column vector convention."""
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rigid_transform(
    rotation: npt.NDArray[np.float64], translation: list[float]
) -> npt.NDArray[np.float64]:
    """Assemble a 4x4 homogeneous matrix from a rotation and a translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array(translation, dtype=np.float64)
    return transform


def homogeneous_intrinsics(intrinsics: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pad per-camera 3x3 intrinsics to 4x4 homogeneous matrices."""
    homogeneous = np.tile(np.eye(4, dtype=np.float64), (intrinsics.shape[0], 1, 1))
    homogeneous[:, :3, :3] = intrinsics
    return homogeneous


def homogeneous_pixel_affines(pixel_affines: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lift per-camera 3x3 pixel affines into the 4x4 layout of image_augmentation_matrices."""
    homogeneous = np.tile(np.eye(4, dtype=np.float64), (pixel_affines.shape[0], 1, 1))
    homogeneous[:, :2, :2] = pixel_affines[:, :2, :2]
    homogeneous[:, :2, 3] = pixel_affines[:, :2, 2]
    return homogeneous


def augmentation_matrix(
    yaw: float, scale: float, translation: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compose the 4x4 augmentation the transforms save: rotation, then scale, then translation.

    Args:
        yaw: Rotation around the z axis in radians.
        scale: Uniform scale factor.
        translation: Translation vector with three entries.

    Returns:
        The 4x4 matrix in column vector convention.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.multiply(scale, rotation_about_z(yaw))
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def to_float64(tensor: Tensor) -> npt.NDArray[np.float64]:
    """Convert a tensor to a float64 numpy array for the expected value computations."""
    return tensor.detach().cpu().numpy().astype(np.float64)


def to_float32_tensor(array: npt.NDArray[np.float64]) -> Tensor:
    """Convert a numpy array to the float32 tensor the containers expect."""
    return torch.tensor(array, dtype=torch.float32)


def project_to_image(
    lidar2images: npt.NDArray[np.float64], xyz: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Project lidar points through every camera's lidar to image matrix.

    Args:
        lidar2images: Per-camera 4x4 lidar to image matrices, shape (num_cameras, 4, 4).
        xyz: Points to project, shape (num_points, 3).

    Returns:
        The homogeneous image coordinates, shape (num_cameras, num_points, 4).
    """
    homogeneous_xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    return np.einsum("cij,pj->cpi", lidar2images, homogeneous_xyz)


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

    def build_camera_intrinsics(self) -> npt.NDArray[np.float64]:
        """Build the raw intrinsics of a wide front camera and a narrower side camera.

        Returns:
            Per-camera 3x3 intrinsics, shape (2, 3, 3).
        """
        return np.array(
            [
                [[1200.0, 0.0, 960.0], [0.0, 1180.0, 540.0], [0.0, 0.0, 1.0]],
                [[800.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )

    def build_pixel_affines(self) -> npt.NDArray[np.float64]:
        """Build the image-space augmentations already applied to the images.

        The front camera is resized by a half, the side camera by a quarter and then cropped
        by 40 rows, so the augmented intrinsics differ from the raw ones per camera.

        Returns:
            Per-camera 3x3 pixel affines, shape (2, 3, 3).
        """
        return np.array(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                [[0.25, 0.0, 0.0], [0.0, 0.25, -40.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )

    def build_lidar2cams(self) -> npt.NDArray[np.float64]:
        """Build the lidar to camera extrinsics of a front camera and a left camera.

        Both carry a distinct rotation (the optical axis permutation with a small
        misalignment, and a yaw of 1.5 rad for the left one) and a distinct translation.

        Returns:
            Per-camera 4x4 lidar to camera matrices, shape (2, 4, 4).
        """
        front_rotation = LIDAR_TO_OPTICAL_AXES @ rotation_about_y(0.05) @ rotation_about_x(-0.02)
        left_rotation = LIDAR_TO_OPTICAL_AXES @ rotation_about_z(-1.5) @ rotation_about_x(0.1)
        return np.stack(
            [
                rigid_transform(front_rotation, [0.3, -0.1, -0.8]),
                rigid_transform(left_rotation, [-0.6, 0.4, -0.5]),
            ]
        )

    def build_camera_image_data(self) -> BaseImages:
        """Build two cameras with non-trivial, per-camera intrinsics and extrinsics.

        ``lidar2images`` follows the loading contract: the augmented intrinsics applied after
        the lidar to camera matrices.

        Returns:
            BaseImages holding two cameras.
        """
        camera_intrinsics = self.build_camera_intrinsics()
        pixel_affines = self.build_pixel_affines()
        augmented_camera_intrinsics = pixel_affines @ camera_intrinsics
        lidar2cams = self.build_lidar2cams()
        lidar2images = homogeneous_intrinsics(augmented_camera_intrinsics) @ lidar2cams
        fields = {
            "images": torch.ones((2, 3, 4, 4), dtype=torch.float32),
            "timestamps": torch.tensor([0.0, 0.05], dtype=torch.float64),
            "camera_intrinsics": to_float32_tensor(camera_intrinsics),
            "camera_names": ["camera_front", "camera_left"],
            "lidar2images": to_float32_tensor(lidar2images),
            "lidar2cams": to_float32_tensor(lidar2cams),
            "distortion_models": ["plumb_bob", "plumb_bob"],
            "distortion_coefficients": [torch.zeros(5), torch.zeros(5)],
            "augmented_camera_intrinsics": to_float32_tensor(augmented_camera_intrinsics),
            "image_augmentation_matrices": to_float32_tensor(
                homogeneous_pixel_affines(pixel_affines)
            ),
        }
        return BaseImages.model_validate(fields)

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

    def snapshot_camera_image_data(self, camera_image_data: BaseImages) -> None:
        """Keep float64 copies of the camera fixtures the expected values are computed from.

        Args:
            camera_image_data: The camera data before any transform runs.
        """
        self.original_camera_image_data = camera_image_data
        self.original_lidar2cams = to_float64(camera_image_data.lidar2cams)
        self.original_lidar2images = to_float64(camera_image_data.lidar2images)
        self.original_augmented_camera_intrinsics = to_float64(
            camera_image_data.augmented_camera_intrinsics
        )

    def expected_camera_matrices(
        self, transformation_matrix: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compose the camera matrices the fixtures must hold after a lidar augmentation.

        The cameras do not move, so the lidar to camera matrices absorb the inverse of the
        augmentation, and the lidar to image matrices apply the augmented intrinsics after
        them.

        Args:
            transformation_matrix: The 4x4 augmentation applied to the lidar frame.

        Returns:
            The expected per-camera lidar to camera and lidar to image matrices.
        """
        expected_lidar2cams = self.original_lidar2cams @ np.linalg.inv(transformation_matrix)
        expected_lidar2images = (
            homogeneous_intrinsics(self.original_augmented_camera_intrinsics) @ expected_lidar2cams
        )
        return expected_lidar2cams, expected_lidar2images

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
        # The upper-left block is the scaled rotation, so any of its columns has the scale as norm.
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
                + math.pi,
                math.tau,
            )
            - math.pi
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

    def assert_matrices_close(
        self, matrices: Tensor, expected_matrices: npt.NDArray[np.float64]
    ) -> None:
        """Assert that float32 matrices match float64 expected ones up to float32 rounding.

        Args:
            matrices: The matrices under test.
            expected_matrices: The matrices they are expected to match.
        """
        np.testing.assert_allclose(to_float64(matrices), expected_matrices, rtol=1e-5, atol=1e-4)

    def assert_camera_matrices_follow(
        self,
        camera_image_data: BaseImages,
        transformation_matrix: npt.NDArray[np.float64],
    ) -> None:
        """Assert that the camera matrices were re-expressed in the augmented lidar frame.

        Args:
            camera_image_data: The camera data after the transform.
            transformation_matrix: The 4x4 augmentation the transform applied to the lidar.
        """
        expected_lidar2cams, expected_lidar2images = self.expected_camera_matrices(
            transformation_matrix
        )
        self.assert_matrices_close(camera_image_data.lidar2cams, expected_lidar2cams)
        self.assert_matrices_close(camera_image_data.lidar2images, expected_lidar2images)
        self.assert_camera_calibration_untouched(camera_image_data)

    def assert_camera_calibration_untouched(self, camera_image_data: BaseImages) -> None:
        """Assert that a lidar augmentation left the per-camera calibration and images alone.

        Args:
            camera_image_data: The camera data after the transform.
        """
        original = self.original_camera_image_data
        self.assertTrue(torch.equal(camera_image_data.images, original.images))
        self.assertTrue(torch.equal(camera_image_data.timestamps, original.timestamps))
        self.assertEqual(list(camera_image_data.camera_names), list(original.camera_names))
        self.assertTrue(
            torch.equal(camera_image_data.camera_intrinsics, original.camera_intrinsics)
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                original.augmented_camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                original.image_augmentation_matrices,
            )
        )

    def assert_projections_preserved(
        self,
        camera_image_data: BaseImages,
        original_xyz: npt.NDArray[np.float64],
        transformed_xyz: npt.NDArray[np.float64],
    ) -> None:
        """Assert that the transformed points project exactly where the originals did.

        The cameras never move and the images are never re-rendered, so a point augmented in
        the lidar frame has to land on the same pixel at the same depth through the updated
        lidar to image matrices as the original point did through the original ones.

        Args:
            camera_image_data: The camera data after the transform.
            original_xyz: The points before the transform, shape (num_points, 3).
            transformed_xyz: The same points after the transform, shape (num_points, 3).
        """
        expected_image_points = project_to_image(self.original_lidar2images, original_xyz)
        image_points = project_to_image(to_float64(camera_image_data.lidar2images), transformed_xyz)
        np.testing.assert_allclose(image_points, expected_image_points, rtol=1e-5, atol=1e-3)
        # The pixel coordinates are the first two entries divided by the depth.
        expected_pixels = expected_image_points[:, :, :2] / expected_image_points[:, :, 2:3]
        pixels = image_points[:, :, :2] / image_points[:, :, 2:3]
        np.testing.assert_allclose(pixels, expected_pixels, rtol=1e-5, atol=1e-3)


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
        self.snapshot_camera_image_data(self.sample.camera_image_data)
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

    def replay_random_draw(self, seed: int) -> npt.NDArray[np.float64]:
        """Replay the yaw, scale and translation the transform draws from a seed.

        The draw is reproduced with numpy alone, in the order the transform documents, so the
        expected augmentation does not depend on the transform's own matrix construction.

        Args:
            seed: The seed the transform is run with.

        Returns:
            The 4x4 augmentation the transform is expected to save.
        """
        np.random.seed(seed)
        yaw = float(np.random.uniform(self.yaw_rot_range[0], self.yaw_rot_range[1]))
        scale = float(np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1]))
        translation = np.random.normal(0.0, np.array(self.translation_std), size=(1, 3))
        return augmentation_matrix(yaw, scale, translation)

    def run_seeded_transform(
        self, sample: ModelGTSample, seed: int = 0
    ) -> tuple[ModelGTSample, npt.NDArray[np.float64]]:
        """Run the randomized transform from a seed alongside the independently replayed draw.

        Args:
            sample: The sample to transform.
            seed: The seed the transform is run with.

        Returns:
            The transformed sample and the 4x4 augmentation it is expected to have applied.
        """
        expected_transformation_matrix = self.replay_random_draw(seed)
        np.random.seed(seed)
        output = self.build_transform()(sample)
        return output, expected_transformation_matrix

    def test_saves_transformation_matrix(self) -> None:
        """Test that the sampled augmentation is saved as the expected 4x4 matrix."""
        output, expected_transformation_matrix = self.run_seeded_transform(self.sample)

        assert output.lidar_transformation_sample is not None
        lidar_transformation_sample = output.lidar_transformation_sample
        self.assert_matrices_close(
            lidar_transformation_sample.transformation_matrix, expected_transformation_matrix
        )
        # The replayed draw is a genuine augmentation, not an identity the test would miss.
        self.assertFalse(np.allclose(expected_transformation_matrix, np.eye(4), atol=1e-3))
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
        """Test that the camera matrices are re-expressed in the augmented frame."""
        output, expected_transformation_matrix = self.run_seeded_transform(self.sample)

        assert output.camera_image_data is not None
        self.assert_camera_matrices_follow(output.camera_image_data, expected_transformation_matrix)

    def test_rotation_and_scale_update_the_extrinsic_rotation_block_only(self) -> None:
        """Test a fixed yaw and scale against the closed form of the extrinsics update.

        Without a translation the inverse augmentation is the inverse rotation divided by
        the scale, so the rotation block of every extrinsic is rotated back and shrunk while
        the camera positions stay where they were.
        """
        output = GlobalRotScaleTrans(yaw_rot_range=[0.3, 0.3], scale_ratio_range=[2.0, 2.0])(
            self.sample
        )

        assert output.camera_image_data is not None
        lidar2cams = to_float64(output.camera_image_data.lidar2cams)
        expected_rotation_blocks = np.divide(
            self.original_lidar2cams[:, :3, :3] @ rotation_about_z(-0.3), 2.0
        )
        np.testing.assert_allclose(
            lidar2cams[:, :3, :3], expected_rotation_blocks, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            lidar2cams[:, :3, 3], self.original_lidar2cams[:, :3, 3], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            lidar2cams[:, 3, :], np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        )
        # The two cameras keep distinct extrinsics, so a per-camera mix-up would show.
        self.assertFalse(np.allclose(lidar2cams[0], lidar2cams[1], atol=1e-3))
        self.assert_camera_matrices_follow(
            output.camera_image_data, augmentation_matrix(0.3, 2.0, np.zeros(3))
        )

    def test_translation_updates_the_extrinsic_translation_only(self) -> None:
        """Test a pure translation against the closed form of the extrinsics update.

        The inverse of a translation by t is a translation by -t, so every rotation block is
        unchanged and every camera position moves by the rotation block applied to -t.
        """
        self.yaw_rot_range = [0.0, 0.0]
        self.scale_ratio_range = [1.0, 1.0]
        output, expected_transformation_matrix = self.run_seeded_transform(self.sample)

        assert output.camera_image_data is not None
        translation = expected_transformation_matrix[:3, 3]
        self.assertFalse(np.allclose(translation, np.zeros(3), atol=1e-3))
        lidar2cams = to_float64(output.camera_image_data.lidar2cams)
        np.testing.assert_allclose(
            lidar2cams[:, :3, :3], self.original_lidar2cams[:, :3, :3], rtol=1e-5, atol=1e-6
        )
        expected_positions = self.original_lidar2cams[:, :3, 3] - np.einsum(
            "cij,j->ci", self.original_lidar2cams[:, :3, :3], translation
        )
        np.testing.assert_allclose(lidar2cams[:, :3, 3], expected_positions, rtol=1e-5, atol=1e-5)
        self.assert_camera_matrices_follow(output.camera_image_data, expected_transformation_matrix)

    def test_lidar2images_uses_the_augmented_intrinsics(self) -> None:
        """Test that lidar2images is rebuilt from the augmented, not the raw, intrinsics."""
        output, expected_transformation_matrix = self.run_seeded_transform(self.sample)

        assert output.camera_image_data is not None
        expected_lidar2cams, _ = self.expected_camera_matrices(expected_transformation_matrix)
        raw_intrinsics_lidar2images = (
            homogeneous_intrinsics(to_float64(self.original_camera_image_data.camera_intrinsics))
            @ expected_lidar2cams
        )
        lidar2images = to_float64(output.camera_image_data.lidar2images)
        self.assertFalse(np.allclose(lidar2images, raw_intrinsics_lidar2images, atol=1e-3))
        np.testing.assert_allclose(
            lidar2images,
            homogeneous_intrinsics(self.original_augmented_camera_intrinsics) @ expected_lidar2cams,
            rtol=1e-5,
            atol=1e-4,
        )

    def test_transformed_points_project_onto_the_same_pixels(self) -> None:
        """Test that the augmented points keep projecting onto the original pixels."""
        output = self.build_transform()(self.sample)

        assert output.point_cloud_data is not None
        assert output.camera_image_data is not None
        transformed_xyz = to_float64(output.point_cloud_data.points[:, :3])
        self.assertFalse(np.allclose(transformed_xyz, to_float64(self.original_points[:, :3])))
        self.assert_projections_preserved(
            output.camera_image_data, to_float64(self.original_points[:, :3]), transformed_xyz
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
            torch.allclose(
                bbox_params[:, 3:6], torch.mul(self.original_bbox_params[:, 3:6], 2.0), atol=1e-5
            )
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
        """
        Test that a camera-only sample, as a camera-only pipeline serves, updates the cameras
        and leaves the points None: a point the sample would have held still projects where it
        did.
        """
        sample = self.build_multi_task_gt_sample(with_point_cloud_data=False)
        output, expected_transformation_matrix = self.run_seeded_transform(sample)

        assert output.camera_image_data is not None
        self.assertIsNone(output.point_cloud_data)
        self.assert_camera_matrices_follow(output.camera_image_data, expected_transformation_matrix)
        original_xyz = to_float64(self.original_points[:, :3])
        transformed_xyz = to_float64(
            self.apply_transformation_to_xyz(
                self.original_points[:, :3], to_float32_tensor(expected_transformation_matrix)
            )
        )
        self.assert_projections_preserved(output.camera_image_data, original_xyz, transformed_xyz)

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
        assert second_output.camera_image_data is not None
        self.assertTrue(
            torch.allclose(
                second_output.lidar_transformation_sample.transformation_matrix,
                first_transformation_matrix @ first_transformation_matrix,
                atol=1e-5,
            )
        )
        single_order = [
            TransformationName.ROTATION,
            TransformationName.SCALING,
            TransformationName.TRANSLATION,
        ]
        self.assertEqual(
            second_output.lidar_transformation_sample.transformation_order,
            single_order + single_order,
        )
        # The cameras absorb both augmentations, i.e. a yaw of 0.6 and a scale of 4.
        self.assert_camera_matrices_follow(
            second_output.camera_image_data, augmentation_matrix(0.6, 4.0, np.zeros(3))
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
        """Test that composing the camera matrices with the augmentation restores them."""
        output, expected_transformation_matrix = self.run_seeded_transform(self.sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        lidar2cams = to_float64(camera_image_data.lidar2cams)
        lidar2images = to_float64(camera_image_data.lidar2images)
        self.assertFalse(np.allclose(lidar2cams, self.original_lidar2cams, atol=1e-3))
        self.assertFalse(np.allclose(lidar2images, self.original_lidar2images, atol=1e-3))

        # The matrices were composed with the inverse of the augmentation, so composing them
        # with the augmentation itself brings them back.
        np.testing.assert_allclose(
            lidar2cams @ expected_transformation_matrix,
            self.original_lidar2cams,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            lidar2images @ expected_transformation_matrix,
            self.original_lidar2images,
            rtol=1e-5,
            atol=1e-3,
        )

    def test_sample_without_points_or_cameras_still_moves_the_boxes(self) -> None:
        """
        Test that a sample with neither points nor cameras is transformed: the boxes move and
        the augmentation is recorded, so no modality is required.
        """
        sample = self.build_multi_task_gt_sample(
            with_point_cloud_data=False, with_camera_image_data=False
        )
        output, expected_transformation_matrix = self.run_seeded_transform(sample)

        self.assertIsNone(output.point_cloud_data)
        self.assertIsNone(output.camera_image_data)
        assert output.lidar_transformation_sample is not None
        np.testing.assert_allclose(
            to_float64(output.lidar_transformation_sample.transformation_matrix),
            expected_transformation_matrix,
            rtol=1e-5,
            atol=1e-5,
        )
        assert output.detection3d_gt_bboxes_3d is not None
        self.assertFalse(
            torch.equal(output.detection3d_gt_bboxes_3d.bbox_params, self.original_bbox_params)
        )


class TestGlobalRotScaleTransCameraOnly(PinholeCameraTestCase):
    """The global 3D augmentation of a camera-only sample, as a camera-only pipeline serves.

    The sample holds the images, their calibration and the boxes but no points, which
    ``TestGlobalRotScaleTrans.test_camera_only_sample`` covers for the calibration alone. These
    tests follow the boxes and the 2D targets derived from them, and the folding of the
    augmentation into the ego pose a temporal model relies on.
    """

    def build_camera_sample(self) -> ModelGTSample:
        """A camera-only sample with one box in front of a forward-looking camera."""
        images = self.build_pinhole_images([self.LIDAR2CAM_FORWARD], 160, 96)
        return self.build_sample(images)._replace(
            detection3d_gt_bboxes_3d=self.build_boxes([[10.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.3]], [0])
        )

    def augment(self, sample: ModelGTSample, seed: int) -> ModelGTSample:
        """Apply a random rotation, scale and translation."""
        np.random.seed(seed)
        return GlobalRotScaleTrans(
            yaw_rot_range=[-0.5, 0.5],
            scale_ratio_range=[0.9, 1.1],
            translation_std=[0.5, 0.5, 0.2],
        )(sample)

    def test_keeps_the_2d_targets(self) -> None:
        """
        Input: the 2D targets projected from a camera-only sample before and after a random
        rotation, scale and translation.
        Expected: the boxes and the calibration move together, so the projection, box edges
        and projected center alike, does not change.
        Check: the labels are equal and the box rows agree within 1e-3 pixels.
        """
        sample = self.build_camera_sample()
        before = LoadAnnotations2DFromBoxes3D()(sample).detection2d_gt_sample

        after = LoadAnnotations2DFromBoxes3D()(self.augment(sample, seed=1)).detection2d_gt_sample

        self.assertEqual(before.gt_labels_2d[0].tolist(), after.gt_labels_2d[0].tolist())
        torch.testing.assert_close(before.gt_bboxes_2d[0], after.gt_bboxes_2d[0], atol=1e-3, rtol=0)

    def test_frames_augmented_on_their_own_stay_aligned_through_the_folded_poses(self) -> None:
        """
        Input: one parked car seen from two ego poses, every frame augmented with its own
        random rotation, scale and translation.
        Expected: the augmentation moves the boxes of a sample without points and is recorded,
        and the frame metadata folds it into the pose, so a temporal model maps the box of the
        previous frame onto the box of the current one through the folded poses alone.
        Check: every folded pose differs from the raw one, so the alignment is not that of
        unaugmented frames, and the previous box center mapped through both folded poses lands
        on the current box center within 1 cm.
        """
        global_box = torch.tensor([3012.0, -1180.0, 0.5, 1.0])
        poses = []
        for translation, yaw in (([3000.0, -1200.0, 0.0], 0.3), ([3002.0, -1198.5, 0.0], 0.35)):
            pose = torch.eye(4)
            pose[:2, :2] = torch.tensor(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            pose[:3, 3] = torch.tensor(translation)
            poses.append(pose)
        augmented = []
        for seed, pose in enumerate(poses):
            local_box = (torch.linalg.inv(pose) @ global_box)[:3]
            images = self.build_pinhole_images([self.LIDAR2CAM_FORWARD], 160, 96)
            sample = self.build_sample(images)._replace(
                detection3d_gt_bboxes_3d=self.build_boxes(
                    [[*local_box.tolist(), 4.0, 2.0, 1.5, 0.0]], [0]
                ),
                frame_meta=FrameMetaSample(ego2global=pose, scene_token="scene"),
            )
            augmented.append(self.augment(sample, seed=seed))
        folded = FrameMetaBatch.collate_gt_samples(
            [sample.frame_meta for sample in augmented],
            [sample.lidar_transformation_sample for sample in augmented],
        ).ego2globals
        for raw, folded_pose in zip(poses, folded, strict=True):
            self.assertFalse(torch.allclose(folded_pose, raw, atol=1e-3))

        previous_center = torch.cat(
            [augmented[0].detection3d_gt_bboxes_3d.center[0], torch.ones(1)]
        )
        mapped = torch.linalg.inv(folded[1]) @ folded[0] @ previous_center

        torch.testing.assert_close(
            mapped[:3], augmented[1].detection3d_gt_bboxes_3d.center[0], atol=1e-2, rtol=0
        )


class TestRandomRotateTargetAngle(BaseCameraLidarGeometryTestCase):
    """Unit tests for the camera-lidar RandomRotateTargetAngle transform."""

    def setUp(self) -> None:
        """Set up a lidar-only sample for all tests."""
        self.sample = self.build_multi_task_gt_sample(
            with_camera_image_data=False, with_bboxes_3d=False
        )
        assert self.sample.point_cloud_data is not None
        self.original_points = self.sample.point_cloud_data.points.clone()

    def test_rotates_by_the_only_target_angle(self) -> None:
        """Test that a single target of pi radians negates the x and y coordinates."""
        output = RandomRotateTargetAngle(probability=1.0, yaw_angle_ratios=[1.0])(self.sample)

        assert output.point_cloud_data is not None
        expected = self.original_points.clone()
        expected[:, :2] = -expected[:, :2]
        torch.testing.assert_close(output.point_cloud_data.points, expected, atol=1e-6, rtol=0.0)
        self.assertIsNotNone(output.lidar_transformation_sample)

    def test_skips_the_rotation_at_zero_probability(self) -> None:
        """Test that the transform leaves the points untouched when it is never applied."""
        output = RandomRotateTargetAngle(probability=0.0, yaw_angle_ratios=[1.0])(self.sample)

        assert output.point_cloud_data is not None
        torch.testing.assert_close(output.point_cloud_data.points, self.original_points)


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
        self.snapshot_camera_image_data(self.sample.camera_image_data)
        self.horizontal_flip = np.diag([1.0, -1.0, 1.0, 1.0])
        self.vertical_flip = np.diag([-1.0, 1.0, 1.0, 1.0])
        self.both_flips = np.diag([-1.0, -1.0, 1.0, 1.0])

    def test_flips_points_bboxes_and_camera_matrices(self) -> None:
        """Test that both BEV flips negate x and y for the points, bboxes and cameras."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(
            self.sample
        )

        assert output.lidar_transformation_sample is not None
        assert output.point_cloud_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        assert output.camera_image_data is not None
        lidar_transformation_sample = output.lidar_transformation_sample
        self.assert_matrices_close(
            lidar_transformation_sample.transformation_matrix, self.both_flips
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
        self.assert_camera_matrices_follow(output.camera_image_data, self.both_flips)

    def test_flip_negates_the_extrinsic_columns(self) -> None:
        """Test a horizontal flip against the closed form of the extrinsics update.

        A flip is its own inverse, so the lidar to camera matrices only negate the column
        of the flipped axis while the other columns, including the camera positions, stay.
        """
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.camera_image_data is not None
        lidar2cams = to_float64(output.camera_image_data.lidar2cams)
        expected_lidar2cams = self.original_lidar2cams.copy()
        expected_lidar2cams[:, :, 1] = -expected_lidar2cams[:, :, 1]
        np.testing.assert_allclose(lidar2cams, expected_lidar2cams, rtol=1e-6, atol=1e-6)
        self.assert_camera_matrices_follow(output.camera_image_data, self.horizontal_flip)

    def test_flipped_points_project_onto_the_same_pixels(self) -> None:
        """Test that the flipped points keep projecting onto the original pixels."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(
            self.sample
        )

        assert output.point_cloud_data is not None
        assert output.camera_image_data is not None
        self.assert_projections_preserved(
            output.camera_image_data,
            to_float64(self.original_points[:, :3]),
            to_float64(output.point_cloud_data.points[:, :3]),
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
        self.assert_matrices_close(output.camera_image_data.lidar2cams, self.original_lidar2cams)
        self.assert_matrices_close(
            output.camera_image_data.lidar2images, self.original_lidar2images
        )
        self.assert_camera_calibration_untouched(output.camera_image_data)

    def test_horizontal_flip_only_negates_lateral_axis(self) -> None:
        """Test that a horizontal flip mirrors the lateral (y) axis only."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.lidar_transformation_sample is not None
        assert output.point_cloud_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        self.assert_matrices_close(
            output.lidar_transformation_sample.transformation_matrix, self.horizontal_flip
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
        """
        Test that a camera-only sample flips its boxes and camera matrices and leaves the
        points None.
        """
        sample = self.build_multi_task_gt_sample(with_point_cloud_data=False)
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(sample)

        assert output.camera_image_data is not None
        assert output.detection3d_gt_bboxes_3d is not None
        self.assertIsNone(output.point_cloud_data)
        self.assert_camera_matrices_follow(output.camera_image_data, self.both_flips)
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
        assert output.camera_image_data is not None
        # Flipping twice along the same axis returns to the identity.
        self.assertTrue(
            torch.allclose(output.lidar_transformation_sample.transformation_matrix, torch.eye(4))
        )
        self.assertEqual(
            output.lidar_transformation_sample.transformation_order,
            [TransformationName.HORIZONTAL_FLIP, TransformationName.HORIZONTAL_FLIP],
        )
        self.assertTrue(torch.allclose(output.point_cloud_data.points, self.original_points))
        self.assert_camera_matrices_follow(output.camera_image_data, np.eye(4))

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
        """Test that composing the camera matrices with the flip restores them."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        lidar2cams = to_float64(camera_image_data.lidar2cams)
        lidar2images = to_float64(camera_image_data.lidar2images)
        self.assertFalse(np.allclose(lidar2cams, self.original_lidar2cams, atol=1e-3))
        self.assertFalse(np.allclose(lidar2images, self.original_lidar2images, atol=1e-3))

        # The matrices were composed with the inverse of the flip, so composing them with the
        # flip itself brings them back.
        np.testing.assert_allclose(
            lidar2cams @ self.horizontal_flip, self.original_lidar2cams, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            lidar2images @ self.horizontal_flip, self.original_lidar2images, rtol=1e-6, atol=1e-4
        )

    def test_sample_without_points_or_cameras_still_flips_the_boxes(self) -> None:
        """
        Test that a sample with neither points nor cameras is flipped: the boxes move and the
        flip is recorded, so no modality is required.
        """
        sample = self.build_multi_task_gt_sample(
            with_point_cloud_data=False, with_camera_image_data=False
        )
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(sample)

        self.assertIsNone(output.point_cloud_data)
        self.assertIsNone(output.camera_image_data)
        assert output.detection3d_gt_bboxes_3d is not None
        self.assertTrue(
            torch.allclose(
                output.detection3d_gt_bboxes_3d.bbox_params[:, :2],
                -self.original_bbox_params[:, :2],
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
