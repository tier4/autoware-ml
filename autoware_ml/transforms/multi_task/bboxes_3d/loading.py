"""
Bboxes 3d transforms for loading bboxes (for example, label name filter).
The code is modified based on https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from typing import Sequence

import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform


class BBoxesLabelNameFilter(MultiTaskBaseTransform):
    """Filter 3D bounding boxes by label names."""

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, label_names_to_keep: Sequence[str]) -> None:
        """Initialize the BBoxesLabelNameFilter transform."""
        super().__init__(probability=None)
        self.label_names_to_keep = label_names_to_keep

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter 3D bounding boxes by label names."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        bboxes_to_keep_mask = torch.tensor(
            [
                True if label_name in self.label_names_to_keep else False
                for label_name in detection3d_gt_bboxes_3d.bbox_label_names
            ],
            dtype=torch.bool,
        )

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return multi_task_gt_sample
