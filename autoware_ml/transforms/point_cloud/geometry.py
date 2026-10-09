"""
Point cloud geometry transforms for augmentation to both points and 3D bboxes
(rotation/scale/translation and BEV flips).
The code is modified based on
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from collections.abc import Sequence
import math

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.spatial import BEVDirection
from autoware_ml.types.geometry import TransformationName

# Flip matrices in column vector convention, keyed by the BEV direction they flip along
BEV_FLIP_MATRICES = {
    BEVDirection.HORIZONTAL: torch.diag(torch.tensor([1.0, -1.0, 1.0])),
    BEVDirection.VERTICAL: torch.diag(torch.tensor([-1.0, 1.0, 1.0])),
}


def yaw_rotation_matrix(angle: float) -> Float32[Tensor, "3 3"]:
    """Build the rotation around the z axis in column vector convention.

    Args:
        angle: Rotation in radians.

    Returns:
        Float32[Tensor, "3 3"]: The rotation matrix.
    """
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def apply_lidar_transformation(
    model_gt_sample: ModelGTSample, lidar_transformation_sample: LiDARTransformationSample
) -> ModelGTSample:
    """Record an augmentation applied to the points of a sample.

    The points moved but the cameras did not, so the lidar to camera matrices take the inverse
    of the augmentation and keep projecting the points onto the same pixels. The augmentation
    is composed after the ones the sample already carries.

    Args:
        model_gt_sample: Sample whose points were transformed.
        lidar_transformation_sample: The augmentation applied to the points.

    Returns:
        ModelGTSample: The sample with the updated camera calibration and transformation.
    """
    camera_image_data = model_gt_sample.camera_image_data
    if camera_image_data is not None:
        camera_image_data = camera_image_data.update_lidar_transformation_matrices(
            torch.linalg.inv(lidar_transformation_sample.transformation_matrix)
        )
    if model_gt_sample.lidar_transformation_sample is not None:
        lidar_transformation_sample = (
            lidar_transformation_sample.create_composed_lidar_transformation_sample(
                previous_lidar_transformation_sample=model_gt_sample.lidar_transformation_sample
            )
        )
    return model_gt_sample._replace(
        lidar_transformation_sample=lidar_transformation_sample,
        camera_image_data=camera_image_data,
    )


class GlobalRotScaleTrans(BaseTransform):
    """Apply global rotation, scaling, and optional translation to point clouds and bboxes."""

    _required_keys = ["point_cloud_data"]

    def __init__(
        self,
        yaw_rot_range: Sequence[float],
        scale_ratio_range: Sequence[float],
        translation_std: Sequence[float] | None = None,
        probability: float | None = None,
    ) -> None:
        """Initialize the GlobalRotScaleTrans transform.

        Args:
            yaw_rot_range: Min and max rotation angles in radians around yaw.
            scale_ratio_range: Min and max scale factors.
            translation_std: Optional per-axis Gaussian translation std ``[x, y, z]``.
            probability: Probability of applying the transform, None to always apply it.
        """
        super().__init__(probability=probability)
        self.yaw_rot_range = yaw_rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            torch.tensor(translation_std, dtype=torch.float32)
            if translation_std is not None
            else None
        )

    def sample_yaw(self) -> float:
        """
        Sample the yaw rotation applied to the sample.

        Returns:
            float: Rotation around yaw in radians.
        """
        return float(np.random.uniform(self.yaw_rot_range[0], self.yaw_rot_range[1]))

    def sample_translation(self) -> Float32[Tensor, "1 3"]:
        """
        Sample the translation applied to the sample.

        Returns:
            Float32[Tensor, "1 3"]: Translation vector, zero when no std is configured.
        """
        if self.translation_std is None:
            return torch.zeros((1, 3), dtype=torch.float32)
        translation = np.random.normal(0.0, self.translation_std, size=(1, 3))
        return torch.tensor(translation, dtype=torch.float32)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Rotate, scale, and translate points and bboxes."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )

        rotation_matrix = yaw_rotation_matrix(self.sample_yaw())
        scale_factor = float(
            np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        )
        translation_vector = self.sample_translation()

        # Points and boxes are rows, so they take the transposed rotation
        point_cloud_data.rotate(rotation_matrix.T)
        point_cloud_data.scale(scale_factor)
        point_cloud_data.translate(translation_vector)

        if model_gt_sample.detection3d_gt_bboxes_3d is not None:
            model_gt_sample.detection3d_gt_bboxes_3d.rotate(rotation_matrix.T)
            model_gt_sample.detection3d_gt_bboxes_3d.scale(scale_factor)
            model_gt_sample.detection3d_gt_bboxes_3d.translate(translation_vector)

        return apply_lidar_transformation(
            model_gt_sample,
            LiDARTransformationSample.create_lidar_transformation_sample(
                rotation_matrix=rotation_matrix,
                scale_factor=scale_factor,
                translation_vector=translation_vector,
                transformation_order=[
                    TransformationName.ROTATION,
                    TransformationName.SCALING,
                    TransformationName.TRANSLATION,
                ],
            ),
        )


class RandomRotateTargetAngle(GlobalRotScaleTrans):
    """Rotate the point cloud and the bboxes by one of a few target yaw angles."""

    def __init__(self, probability: float, yaw_angle_ratios: Sequence[float]) -> None:
        """Initialize the RandomRotateTargetAngle transform.

        Args:
            probability: Probability of applying the transform.
            yaw_angle_ratios: Candidate rotations around yaw, in multiples of pi radians.
        """
        super().__init__(
            yaw_rot_range=(0.0, 0.0),
            scale_ratio_range=(1.0, 1.0),
            translation_std=None,
            probability=probability,
        )
        self.yaw_angle_ratios = list(yaw_angle_ratios)

    def sample_yaw(self) -> float:
        """
        Pick one of the target angles instead of drawing from a continuous range.

        Returns:
            float: Rotation around yaw in radians.
        """
        return float(np.random.choice(self.yaw_angle_ratios)) * float(np.pi)


class GlobalBEVRandomFlip(BaseTransform):
    """Globally and randomly flip point clouds and bboxes along the BEV axes."""

    _required_keys = ["point_cloud_data"]

    def __init__(
        self, horizontal_flip_ratio: float = 0.5, vertical_flip_ratio: float = 0.5
    ) -> None:
        """Initialize the GlobalBEVRandomFlip transform.

        Args:
            horizontal_flip_ratio: Ratio of flipping horizontally.
            vertical_flip_ratio: Ratio of flipping vertically.
        """
        super().__init__(probability=None)
        self.horizontal_flip_ratio = horizontal_flip_ratio
        self.vertical_flip_ratio = vertical_flip_ratio

    def sample_flip(self) -> tuple[bool, bool]:
        """
        Sample random horizontal and vertical flips.
        """
        horizontal_flip = np.random.rand() < self.horizontal_flip_ratio
        vertical_flip = np.random.rand() < self.vertical_flip_ratio
        return horizontal_flip, vertical_flip

    def apply_flip(
        self,
        model_gt_sample: ModelGTSample,
        rotation_matrix: Float32[Tensor, "3 3"],
        bev_flip_direction: BEVDirection,
    ) -> Float32[Tensor, "3 3"]:
        """
        Apply the specified flip to the point cloud and bboxes.

        Args:
            model_gt_sample: The ModelGTSample to apply the flip to.
            rotation_matrix: Flips applied to the sample so far.
            bev_flip_direction: The direction of the flip, horizontal (lateral) or vertical
                (longitudinal).

        Returns:
            Float32[Tensor, "3 3"]: The flips applied so far followed by this one.
        """
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )
        point_cloud_data.flip_bev(bev_direction=bev_flip_direction)
        if model_gt_sample.detection3d_gt_bboxes_3d is not None:
            model_gt_sample.detection3d_gt_bboxes_3d.flip_bev(bev_direction=bev_flip_direction)
        return BEV_FLIP_MATRICES[bev_flip_direction] @ rotation_matrix

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Flip points and bboxes along the specified axis."""
        rotation_matrix = torch.eye(3, dtype=torch.float32)
        horizontal_flip, vertical_flip = self.sample_flip()
        transformation_order = []

        if horizontal_flip:
            rotation_matrix = self.apply_flip(
                model_gt_sample, rotation_matrix, bev_flip_direction=BEVDirection.HORIZONTAL
            )
            transformation_order.append(TransformationName.HORIZONTAL_FLIP)

        if vertical_flip:
            rotation_matrix = self.apply_flip(
                model_gt_sample, rotation_matrix, bev_flip_direction=BEVDirection.VERTICAL
            )
            transformation_order.append(TransformationName.VERTICAL_FLIP)

        return apply_lidar_transformation(
            model_gt_sample,
            LiDARTransformationSample.create_lidar_transformation_sample(
                rotation_matrix=rotation_matrix,
                scale_factor=1.0,
                translation_vector=torch.zeros((1, 3), dtype=torch.float32),
                transformation_order=transformation_order,
            ),
        )


class PointsRangeFilter(BaseTransform):
    """Keep the points inside the half-open range [min, max)."""

    _required_keys = ["point_cloud_data"]

    def __init__(self, points_range: tuple[float, float, float, float, float, float]) -> None:
        """Initialize the PointsRangeFilter transform.

        Args:
            points_range: The range of points to keep in the format
                (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__(probability=None)
        self.points_range = torch.tensor(points_range, dtype=torch.float32)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Filter points based on the specified range."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )
        if not len(point_cloud_data):
            return model_gt_sample
        return model_gt_sample.keep_points(point_cloud_data.in_grid_range_3d(self.points_range))


class PointsRandomShuffle(BaseTransform):
    """Randomly shuffle points in the point cloud."""

    _required_keys = ["point_cloud_data"]

    def __init__(
        self,
    ) -> None:
        """Initialize the PointsRandomShuffle transform."""
        super().__init__(probability=None)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Randomly shuffle points in the point cloud."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )

        if not len(point_cloud_data):
            return model_gt_sample

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        permutation = point_cloud_data.shuffle()
        if model_gt_sample.segmentation3d_gt_sample is None:
            return model_gt_sample

        # Follow the permutation with the labels so both stay aligned
        return model_gt_sample._replace(
            segmentation3d_gt_sample=model_gt_sample.segmentation3d_gt_sample.reorder_labels(
                permutation
            )
        )


class CropBoxInner(BaseTransform):
    """Remove the points that fall inside a 3D box, for example the ego vehicle chassis."""

    _required_keys = ["point_cloud_data"]

    def __init__(self, crop_box: Sequence[float]) -> None:
        """Initialize the CropBoxInner transform.

        Args:
            crop_box: Box bounds [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        super().__init__(probability=None)
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = torch.tensor(crop_box, dtype=torch.float32)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Keep only the points outside the configured box."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )
        return model_gt_sample.keep_points(~point_cloud_data.in_range_3d(self.crop_box))
