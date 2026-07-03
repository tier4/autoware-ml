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

"""Camera-only geometric augmentations.

For camera-only detectors (e.g. StreamPETR) there is no point cloud: the scene
is represented by ``gt_boxes`` and the camera projection matrices. These
transforms apply the same global rotation/scale/translation and flip math as
the lidar variants (shared via ``transforms.geometry3d``) to the boxes and the
camera matrices only. They require ``lidar2cam`` and deliberately never touch
points.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms import geometry3d as g3d
from autoware_ml.transforms.base import BaseTransform


class RandomFlip3D(BaseTransform):
    """Random BEV flip for camera-only models (boxes and camera matrices)."""

    _required_keys: list[str] = ["lidar2cam"]

    def __init__(
        self,
        flip_ratio_bev_horizontal: float = 0.5,
        flip_ratio_bev_vertical: float = 0.5,
    ) -> None:
        """Initialize the random flip transform.

        Args:
            flip_ratio_bev_horizontal: Probability of flipping the lateral (y) axis.
            flip_ratio_bev_vertical: Probability of flipping the longitudinal (x) axis.
        """
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply BEV flips to boxes and camera matrices."""
        flip_x, flip_y = g3d.sample_bev_flips(
            self.flip_ratio_bev_horizontal, self.flip_ratio_bev_vertical
        )
        if flip_y:
            g3d.flip_boxes(input_dict, axis=1)
        if flip_x:
            g3d.flip_boxes(input_dict, axis=0)
        flip = g3d.flip_matrix(flip_x, flip_y)
        g3d.update_camera_matrices(input_dict, np.linalg.inv(flip))
        input_dict["bev_flip_matrix"] = flip
        return input_dict


class GlobalRotScaleTrans(BaseTransform):
    """Global rotation/scale/translation for camera-only models (boxes + camera)."""

    _required_keys: list[str] = ["lidar2cam"]

    def __init__(
        self,
        rot_range: Sequence[float],
        scale_ratio_range: Sequence[float],
        translation_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the global rotation/scale/translation transform.

        Args:
            rot_range: Min and max rotation angles in radians (around z).
            scale_ratio_range: Min and max scale factors.
            translation_std: Per-axis Gaussian translation std ``[x, y, z]``.
                When ``None`` no translation is applied.
        """
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            np.asarray(translation_std, dtype=np.float32) if translation_std is not None else None
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Rotate, scale, and translate boxes and camera matrices."""
        rotation, rotation_angle, scale, translation = g3d.sample_rot_scale_trans(
            self.rot_range, self.scale_ratio_range, self.translation_std
        )
        g3d.transform_boxes(input_dict, rotation, rotation_angle, scale, translation)
        augmentation = g3d.rot_scale_trans_matrix(rotation, scale, translation)
        g3d.update_camera_matrices(input_dict, np.linalg.inv(augmentation))
        input_dict["global_aug_matrix"] = augmentation
        return input_dict
