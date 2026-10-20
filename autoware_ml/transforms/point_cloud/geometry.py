"""
Point cloud geometry transforms (range filters, shuffling and ego box cropping).

The global augmentations that also move the camera calibration (``GlobalRotScaleTrans``,
``RandomRotateTargetAngle`` and ``GlobalBEVRandomFlip``) live in
``autoware_ml.transforms.camera_lidar.geometry``.

The code is modified based on
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.base import BaseTransform


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
