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

"""Scene-level point-cloud augmentations.

These transforms operate on raw ``points`` tensors and optionally update
aligned 3D box annotations when they are present in the sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


class RandomFlip3D(BaseTransform):
    """Randomly flip point clouds across BEV axes."""

    _required_keys = ["points"]
    p = None

    def __init__(
        self, flip_ratio_bev_horizontal: float = 0.5, flip_ratio_bev_vertical: float = 0.5
    ) -> None:
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = input_dict["points"].copy()

        if np.random.rand() < self.flip_ratio_bev_horizontal:
            points[:, 1] *= -1.0
            self._flip_boxes(input_dict, horizontal=True)
        if np.random.rand() < self.flip_ratio_bev_vertical:
            points[:, 0] *= -1.0
            self._flip_boxes(input_dict, horizontal=False)

        input_dict["points"] = points
        return input_dict

    def _flip_boxes(self, input_dict: dict[str, Any], horizontal: bool) -> None:
        if "gt_boxes" not in input_dict:
            return

        boxes = input_dict["gt_boxes"].copy()
        if horizontal:
            boxes[:, 1] *= -1.0
            boxes[:, 6] *= -1.0
            if boxes.shape[1] >= 9:
                boxes[:, 8] *= -1.0
        else:
            boxes[:, 0] *= -1.0
            boxes[:, 6] = np.pi - boxes[:, 6]
            if boxes.shape[1] >= 9:
                boxes[:, 7] *= -1.0
        input_dict["gt_boxes"] = boxes


class GlobalRotScaleTrans(BaseTransform):
    """Apply global rotation, scaling, and translation to a point cloud."""

    _required_keys = ["points"]

    def __init__(
        self,
        rot_range: list[float],
        scale_ratio_range: list[float],
        translation_std: list[float],
    ) -> None:
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = np.asarray(translation_std, dtype=np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = input_dict["points"].copy()

        rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)
        rotation_matrix = np.array(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        translation = np.random.normal(0.0, self.translation_std, size=(1, 3)).astype(np.float32)

        points[:, :3] = (points[:, :3] @ rotation_matrix.T) * scale + translation
        input_dict["points"] = points
        self._transform_boxes(input_dict, rotation_matrix, rotation, scale, translation)
        return input_dict

    def _transform_boxes(
        self,
        input_dict: dict[str, Any],
        rotation_matrix: npt.NDArray[np.float32],
        rotation: float,
        scale: float,
        translation: npt.NDArray[np.float32],
    ) -> None:
        if "gt_boxes" not in input_dict:
            return

        boxes = input_dict["gt_boxes"].copy()
        boxes[:, :3] = (boxes[:, :3] @ rotation_matrix.T) * scale + translation
        boxes[:, 3:6] *= scale
        boxes[:, 6] += rotation
        if boxes.shape[1] >= 9:
            boxes[:, 7:9] = boxes[:, 7:9] @ rotation_matrix[:2, :2].T
        input_dict["gt_boxes"] = boxes
