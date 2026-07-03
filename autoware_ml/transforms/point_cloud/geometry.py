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

"""Point-cloud geometric augmentations (lidar-only).

Operate on the point representation (``coord`` and/or ``points``), per-point
``normal`` and ``gt_boxes`` when present. They require a point cloud and never
touch camera matrices - the camera-aware variants live in
``transforms.camera_lidar.geometry`` and ``transforms.camera.geometry`` and
share the exact same math via ``transforms.geometry3d``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms import geometry3d as g3d
from autoware_ml.transforms.base import BaseTransform


class RandomRotateTargetAngle(BaseTransform):
    """Rotate the point cloud by one sampled discrete target angle."""

    _required_keys: list[str] = []

    def __init__(
        self,
        *,
        p: float = 0.5,
        angle: Sequence[float],
        axis: str = "z",
        center: Sequence[float] | None = None,
    ) -> None:
        """Initialize the RandomRotateTargetAngle transform.

        Args:
            p: Probability of applying the transform.
            angle: Candidate rotation angles in multiples of ``pi`` radians.
            axis: Rotation axis. Only ``z`` is supported for box-aware use.
            center: Optional rotation center.
        """
        self.p = p
        self.angle = list(angle)
        self.axis = axis
        self.center = np.asarray(center, dtype=np.float32) if center is not None else None

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Rotate point coordinates by one selected target angle."""
        g3d.require_point_cloud(input_dict)
        angle = float(np.random.choice(self.angle)) * np.pi
        rotation = g3d.rotation_matrix(self.axis, angle)
        reference = input_dict.get("coord")
        if reference is None:
            reference = input_dict.get("points")
        center = g3d.resolve_rotation_center(np.asarray(reference)[:, :3], self.center)
        g3d.rotate_points_about_center(input_dict, rotation, center)
        g3d.transform_normal(input_dict, rotation)
        return input_dict


class RandomFlip3D(BaseTransform):
    """Randomly flip the point cloud (and boxes / normals) across the BEV axes."""

    _required_keys: list[str] = []

    def __init__(
        self,
        *,
        flip_ratio_bev_horizontal: float = 0.5,
        flip_ratio_bev_vertical: float = 0.5,
    ) -> None:
        """Initialize the RandomFlip3D transform.

        Args:
            flip_ratio_bev_horizontal: Probability of flipping the lateral (y) axis.
            flip_ratio_bev_vertical: Probability of flipping the longitudinal (x) axis.
        """
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply BEV flips to points, normals, and boxes."""
        g3d.require_point_cloud(input_dict)
        flip_x, flip_y = g3d.sample_bev_flips(
            self.flip_ratio_bev_horizontal, self.flip_ratio_bev_vertical
        )
        if flip_y:
            g3d.flip_points(input_dict, axis=1)
            g3d.flip_normal(input_dict, axis=1)
            g3d.flip_boxes(input_dict, axis=1)
        if flip_x:
            g3d.flip_points(input_dict, axis=0)
            g3d.flip_normal(input_dict, axis=0)
            g3d.flip_boxes(input_dict, axis=0)
        return input_dict


class GlobalRotScaleTrans(BaseTransform):
    """Apply global rotation, scaling, and optional translation to the point cloud."""

    _required_keys: list[str] = []

    def __init__(
        self,
        *,
        rot_range: Sequence[float],
        scale_ratio_range: Sequence[float],
        translation_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the GlobalRotScaleTrans transform.

        Args:
            rot_range: Min and max rotation angles in radians around z.
            scale_ratio_range: Min and max scale factors.
            translation_std: Optional per-axis Gaussian translation std ``[x, y, z]``.
        """
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            np.asarray(translation_std, dtype=np.float32) if translation_std is not None else None
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Rotate, scale, and translate points, normals, and boxes."""
        g3d.require_point_cloud(input_dict)
        rotation, rotation_angle, scale, translation = g3d.sample_rot_scale_trans(
            self.rot_range, self.scale_ratio_range, self.translation_std
        )
        g3d.transform_points(input_dict, rotation, scale, translation)
        g3d.transform_normal(input_dict, rotation)
        g3d.transform_boxes(input_dict, rotation, rotation_angle, scale, translation)
        return input_dict
