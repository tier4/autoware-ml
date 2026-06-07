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

"""3D bounding-box filter transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform

_BOX_KEYS = ("gt_boxes", "gt_names", "gt_labels", "gt_num_points")


def _filter_present_box_keys(input_dict: dict[str, Any], mask: np.ndarray) -> None:
    """Apply one per-box mask to every present box-aligned annotation key."""
    for key in _BOX_KEYS:
        if key in input_dict:
            input_dict[key] = input_dict[key][mask]


def _count_points_in_rotated_boxes(
    coord: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Count the number of points inside each oriented 3D bounding box.

    Args:
        coord: Point coordinates of shape ``(N, 3)``.
        boxes: Bounding boxes of shape ``(M, 7)`` with columns
            ``[cx, cy, cz, dx, dy, dz, yaw]``.

    Returns:
        Integer array of shape ``(M,)`` with the point count per box.
    """
    counts = np.zeros(len(boxes), dtype=np.int64)
    for i, box in enumerate(boxes):
        cx, cy, cz, dx, dy, dz, yaw = box[:7]
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        # Translate to box center
        delta = coord[:, :3] - np.array([cx, cy, cz], dtype=np.float32)
        # Rotate into box-local frame (around z-axis)
        local_x = delta[:, 0] * cos_yaw - delta[:, 1] * sin_yaw
        local_y = delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
        local_z = delta[:, 2]
        inside = (
            (np.abs(local_x) <= dx / 2.0)
            & (np.abs(local_y) <= dy / 2.0)
            & (np.abs(local_z) <= dz / 2.0)
        )
        counts[i] = inside.sum()
    return counts


class ObjectNameFilter(BaseTransform):
    """Keep only 3D boxes whose class name is in the allowed list.

    Required keys:
        gt_names: Per-box class name array.

    Optional keys:
        gt_boxes: 3D bounding boxes. Filtered when present.
        gt_labels: Per-box label indices. Filtered when present.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_names: Filtered class names.
        gt_boxes: Filtered boxes (when present).
        gt_labels: Filtered labels (when present).
        gt_num_points: Filtered lidar point counts (when present).
    """

    _required_keys = ["gt_names"]

    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = set(classes)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        mask = np.array([n in self.classes for n in input_dict["gt_names"]], dtype=bool)
        _filter_present_box_keys(input_dict, mask)
        return input_dict


class ObjectRangeFilter(BaseTransform):
    """Filter 3D bounding boxes and associated labels by point-cloud range.

    Required keys:
        (none)

    Optional keys:
        gt_boxes: 3D bounding boxes (Nx7 or Nx9). Filtered when present.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_boxes: Filtered boxes (when present).
        gt_names: Filtered class names (when present alongside gt_boxes).
        gt_labels: Filtered labels (when present alongside gt_boxes).
        gt_num_points: Filtered lidar point counts (when present alongside gt_boxes).
    """

    _required_keys: list[str] = []
    _optional_keys = ["gt_boxes"]

    def __init__(self, point_cloud_range: Sequence[float]) -> None:
        """Initialize the object range filter.

        Args:
            point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
        """
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """No defaults needed — transform is a no-op when gt_boxes is absent."""
        pass

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Filter boxes whose centers fall outside the configured range.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if "gt_boxes" not in input_dict:
            return input_dict

        boxes = input_dict["gt_boxes"]
        pcr = self.point_cloud_range
        mask = (
            (boxes[:, 0] >= pcr[0])
            & (boxes[:, 1] >= pcr[1])
            & (boxes[:, 2] >= pcr[2])
            & (boxes[:, 0] <= pcr[3])
            & (boxes[:, 1] <= pcr[4])
            & (boxes[:, 2] <= pcr[5])
        )
        _filter_present_box_keys(input_dict, mask)
        return input_dict


class ObjectMinPointsFilter(BaseTransform):
    """Remove 3D boxes that contain fewer than a minimum number of points.

    Required keys:
        gt_names: Class name per box.
        coord: Point coordinates (Nx3, float32).

    Optional keys:
        gt_boxes: 3D bounding boxes (Nx7 or Nx9). Filtered when present.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_boxes: Filtered boxes (when present).
        gt_names: Filtered class names.
        gt_labels: Filtered labels (when present).
        gt_num_points: Filtered lidar point counts (when present).
    """

    _required_keys = ["gt_names", "coord"]
    _optional_keys = ["gt_boxes"]

    def __init__(self, min_num_points: int) -> None:
        """Initialize the minimum-points filter.

        Args:
            min_num_points: Minimum number of points required inside each box.
        """
        self.min_num_points = min_num_points

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """No defaults needed — transform is a no-op when gt_boxes is absent."""
        pass

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove boxes with too few interior points.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if "gt_boxes" not in input_dict:
            return input_dict

        coord = np.asarray(input_dict["coord"], dtype=np.float32)
        boxes = input_dict["gt_boxes"]
        counts = _count_points_in_rotated_boxes(coord, boxes)
        mask = counts >= self.min_num_points
        _filter_present_box_keys(input_dict, mask)
        return input_dict
