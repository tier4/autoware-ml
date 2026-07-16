"""
Point cloud geometry transforms for augmentation to both points and 3D bboxes
(rotation/scale/translation and BEV flips).
The code is modified based on https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from typing import Sequence, Tuple

from jaxtyping import Float32
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.datamodule.multi_task.dataclasses.transformation import LiDARTransformationSample
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.transforms.geometry3d import rotation_matrix
from autoware_ml.types.spatial import RotationAxis, BEVDirection
from autoware_ml.types.geometry import TransformationName


class RotationScaleTranslationData(BaseModel):
    """
    Data class to save rotation_matrix, scaling_factor, and translation vector.

    Attributes:
        rotation_matrix: 3x3 rotation matrix.
        scale_factor: Scale factor applied.
        translation_vector: 1x3 translation vector.
    """

    # Set model config to frozen
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # 3x3 rotation matrix, it's saved for column vector convention (left-multiplication), e.g.,
    # R @ points, where points are (3, N) as a column for each dimension.
    rotation_matrix: Float32[Tensor, "3 3"]
    scale_factor: float  # Scale factor applied
    translation_vector: Float32[Tensor, "1 3"]  # Translation vector applied


class GlobalRotScaleTrans(MultiTaskBaseTransform):
    """Apply global rotation, scaling, and optional translation to point clouds and bboxes."""

    _required_keys = ["point_cloud_data"]

    def __init__(
        self,
        yaw_rot_range: Sequence[float],
        scale_ratio_range: Sequence[float],
        translation_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the GlobalRotScaleTrans transform.

        Args:
            yaw_rot_range: Min and max rotation angles in radians around yaw.
            scale_ratio_range: Min and max scale factors.
            translation_std: Optional per-axis Gaussian translation std ``[x, y, z]``.
        """
        super().__init__(probability=None)
        self.yaw_rot_range = yaw_rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            torch.tensor(translation_std, dtype=torch.float32)
            if translation_std is not None
            else None
        )

    def sample_rot_scale_trans(
        self,
    ) -> Tuple[LiDARTransformationSample, RotationScaleTranslationData]:
        """
        Sample random rotation, scale, and translation parameters.
        """
        rotation = float(np.random.uniform(self.yaw_rot_range[0], self.yaw_rot_range[1]))
        matrix = rotation_matrix(str(RotationAxis.Z.name).lower(), rotation)
        scale_factor = float(
            np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        )
        if self.translation_std is not None:
            translation = np.random.normal(0.0, self.translation_std, size=(1, 3)).astype(
                np.float32
            )
        else:
            translation = np.zeros((1, 3), dtype=np.float32)

        # Convert to torch tensor
        rotation_matrix_tensor = torch.tensor(matrix, dtype=torch.float32)
        translation_tensor = torch.tensor(translation, dtype=torch.float32)
        transformation_order = [
            TransformationName.ROTATION,
            TransformationName.SCALING,
            TransformationName.TRANSLATION,
        ]

        rotation_scale_translation_data = RotationScaleTranslationData(
            rotation_matrix=rotation_matrix_tensor,
            scale_factor=scale_factor,
            translation_vector=translation_tensor,
        )
        lidar_transformation_sample = LiDARTransformationSample.create_lidar_transformation_sample(
            rotation_matrix=rotation_matrix_tensor,
            scale_factor=scale_factor,
            translation_vector=translation_tensor,
            transformation_order=transformation_order,
        )
        return lidar_transformation_sample, rotation_scale_translation_data

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Rotate, scale, and translate points and bboxes."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]

        # Sample rotation, scale, and translation parameters
        lidar_transformation_sample, rotation_scale_translation_data = self.sample_rot_scale_trans()

        # Rotate, scale, and translate the point cloud
        # Convert to row vector convention for point cloud transformation
        row_vector_rotation_matrix = rotation_scale_translation_data.rotation_matrix.T
        scale_factor = rotation_scale_translation_data.scale_factor
        translation_vector = rotation_scale_translation_data.translation_vector

        point_cloud_data.rotate(row_vector_rotation_matrix)
        # Scale
        point_cloud_data.scale(scale_factor)
        # Translate
        point_cloud_data.translate(translation_vector)

        # Rotate, scale, and translate the 3D bounding boxes
        if multi_task_gt_sample.detection3d_gt_bboxes_3d is not None:
            multi_task_gt_sample.detection3d_gt_bboxes_3d.rotate(row_vector_rotation_matrix)
            multi_task_gt_sample.detection3d_gt_bboxes_3d.scale(scale_factor)
            multi_task_gt_sample.detection3d_gt_bboxes_3d.translate(translation_vector)

        # Create the composed transformation matrix in the MultiTaskGTSample if it exists
        if multi_task_gt_sample.lidar_transformation_sample is not None:
            lidar_transformation_sample = lidar_transformation_sample.create_composed_lidar_transformation_sample(
                previous_lidar_transformation_sample=multi_task_gt_sample.lidar_transformation_sample
            )

        return MultiTaskGTSample(
            lidar_point_cloud_samples=multi_task_gt_sample.lidar_point_cloud_samples,
            point_cloud_data=multi_task_gt_sample.point_cloud_data,
            detection3d_gt_bboxes_3d=multi_task_gt_sample.detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=multi_task_gt_sample.segmentation3d_gt_sample,
            lidar_transformation_sample=lidar_transformation_sample,
        )


class GlobalBEVRandomFlip(MultiTaskBaseTransform):
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

    def sample_flip(self) -> Tuple[bool, bool]:
        """
        Sample random horizontal and vertical flips.
        """
        horizontal_flip = np.random.rand() < self.horizontal_flip_ratio
        vertical_flip = np.random.rand() < self.vertical_flip_ratio
        return horizontal_flip, vertical_flip

    def apply_flip(
        self,
        multi_task_gt_sample: MultiTaskGTSample,
        rotation_matrix: Float32[Tensor, "3 3"],
        bev_flip_direction: BEVDirection,
    ) -> Float32[Tensor, "3 3"]:
        """
        Apply the specified flip to the point cloud and bboxes.

        Args:
            multi_task_gt_sample: The MultiTaskGTSample to apply the flip to.
            bev_flip_direction: The direction of the flip (horizontal (lateral) or vertical (longitudinal)).
        """
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        if bev_flip_direction == BEVDirection.HORIZONTAL:
            rotation_matrix = (
                torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)
                @ rotation_matrix
            )
        elif bev_flip_direction == BEVDirection.VERTICAL:
            rotation_matrix = (
                torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
                @ rotation_matrix
            )
        else:
            raise ValueError(
                f"Invalid flip direction: {bev_flip_direction}. Must be 'horizontal' or 'vertical'."
            )

        # Flip the point cloud along the direction
        point_cloud_data.flip_bev(bev_direction=bev_flip_direction)

        # Flip the 3D bounding boxes along the direction if they exist
        if multi_task_gt_sample.detection3d_gt_bboxes_3d is not None:
            multi_task_gt_sample.detection3d_gt_bboxes_3d.flip_bev(bev_direction=bev_flip_direction)

        return rotation_matrix

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Flip points and bboxes along the specified axis."""
        rotation_matrix = torch.eye(3, dtype=torch.float32)
        horizontal_flip, vertical_flip = self.sample_flip()
        transformation_order = []

        if horizontal_flip:
            rotation_matrix = self.apply_flip(
                multi_task_gt_sample, rotation_matrix, bev_flip_direction=BEVDirection.HORIZONTAL
            )
            # Add the horizontal flip transformation to the transformation order
            transformation_order.append(TransformationName.HORIZONTAL_FLIP)

        if vertical_flip:
            rotation_matrix = self.apply_flip(
                multi_task_gt_sample, rotation_matrix, bev_flip_direction=BEVDirection.VERTICAL
            )
            # Add the vertical flip transformation to the transformation order
            transformation_order.append(TransformationName.VERTICAL_FLIP)

        # Create the lidar transformation sample
        lidar_transformation_sample = LiDARTransformationSample.create_lidar_transformation_sample(
            rotation_matrix=rotation_matrix,
            scale_factor=1.0,  # No scaling applied
            translation_vector=torch.zeros((1, 3), dtype=torch.float32),  # No translation applied
            transformation_order=transformation_order,
        )

        # Update the lidar transformation sample in the MultiTaskGTSample if it exists
        if multi_task_gt_sample.lidar_transformation_sample is not None:
            lidar_transformation_sample = lidar_transformation_sample.create_composed_lidar_transformation_sample(
                previous_lidar_transformation_sample=multi_task_gt_sample.lidar_transformation_sample
            )

        return MultiTaskGTSample(
            lidar_point_cloud_samples=multi_task_gt_sample.lidar_point_cloud_samples,
            point_cloud_data=multi_task_gt_sample.point_cloud_data,
            detection3d_gt_bboxes_3d=multi_task_gt_sample.detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=multi_task_gt_sample.segmentation3d_gt_sample,
            lidar_transformation_sample=lidar_transformation_sample,
        )


class PointsRangeFilter(MultiTaskBaseTransform):
    """Filter points based on their range."""

    _required_keys = ["point_cloud_data"]

    def __init__(self, points_range: Tuple[float, float, float, float, float, float]) -> None:
        """Initialize the PointsRangeFilter transform.

        Args:
            points_range: The range of points to keep in the format (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__(probability=None)
        self.points_range = torch.tensor(points_range, dtype=torch.float32)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter points based on the specified range."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        if not len(point_cloud_data):
            return multi_task_gt_sample

        point_cloud_range_mask = point_cloud_data.in_range_3d(self.points_range)

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        # TODO(Kok Seang): Need to remove labels outside of range for 3D semantic segmentation.
        point_cloud_data.remove_points(point_cloud_range_mask)
        return multi_task_gt_sample


class PointsRandomShuffle(MultiTaskBaseTransform):
    """Randomly shuffle points in the point cloud."""

    _required_keys = ["point_cloud_data"]

    def __init__(
        self,
    ) -> None:
        """Initialize the PointsRandomShuffle transform."""
        super().__init__(probability=None)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Randomly shuffle points in the point cloud."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]

        if not len(point_cloud_data):
            return multi_task_gt_sample

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        # TODO(Kok Seang): Need to maintain the same order for 3D semantic segmentation.
        point_cloud_data.shuffle()
        return multi_task_gt_sample
