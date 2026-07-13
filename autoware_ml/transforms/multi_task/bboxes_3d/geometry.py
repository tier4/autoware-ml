"""
Bboxes 3d transforms for augmentating bboxes (for example, removing bboxes by distance and number of points).
The code is modified based on https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from typing import Tuple

import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform


class BBoxesMinPointsFilter(MultiTaskBaseTransform):
    """Filter 3D bounding boxes by minimum number of points and distance of bboxes."""

    _required_keys = ["detection3d_gt_bboxes_3d", "point_cloud_data"]

    def __init__(
        self,
        min_points: int,
        bev_range: Tuple[float],
    ) -> None:
        """
        Initialize the BBoxesMinPointsFilter transform.

        Args:
            min_points (int): The minimum number of points required for a bounding box to be kept.
            bev_range (Tuple[float]): The distance ([x_min, y_min, x_max, y_max]) of bounding boxes
                to apply ths mininum number of points filtering.
        """
        super().__init__(probability=None)
        self.min_points = min_points
        self.bev_range = torch.tensor(bev_range, dtype=torch.float32)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter 3D bounding boxes by label names."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]

        distance_in_range_masks = detection3d_gt_bboxes_3d.in_range_bev(self.bev_range)
        points_in_bboxes = detection3d_gt_bboxes_3d.compute_points_in_bboxes(
            points=point_cloud_data.coords,
        )

        # Filter bboxes that are either within the specified distance range and
        # have at least `min_points` points,
        # or are outside the distance range (to keep them).
        keep_bboxes_mask = (
            points_in_bboxes.sum(dim=1) >= self.min_points
        ) & distance_in_range_masks | (~distance_in_range_masks)
        detection3d_gt_bboxes_3d.remove_bboxes(keep_bboxes_mask)
        return multi_task_gt_sample


class BBoxesBEVDistanceFilter(MultiTaskBaseTransform):
    """Filter 3D bounding boxes by their bev distance."""

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(
        self,
        bev_range: Tuple[float],
    ) -> None:
        """
        Initialize the BBoxesBEVDistanceFilter transform.

        Args:
            bev_range (Tuple[float]): The distance ([x_min, y_min, x_max, y_max]) of bounding boxes
                to apply the BEV distance filtering.
        """
        super().__init__(probability=None)
        self.bev_range = torch.tensor(bev_range, dtype=torch.float32)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter 3D bounding boxes by BEV distance."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        distance_in_range_masks = detection3d_gt_bboxes_3d.in_range_bev(self.bev_range)
        detection3d_gt_bboxes_3d.remove_bboxes(distance_in_range_masks)

        return multi_task_gt_sample
