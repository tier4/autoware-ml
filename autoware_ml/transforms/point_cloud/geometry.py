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

"""Point-cloud geometric transforms.

This module provides:
* ``RandomRotateTargetAngle`` - discrete-angle rotation for indoor scenes.
* ``RandomFlip`` - BEV-axis flipping with optional 3D box support.
* ``GlobalRotScaleTrans`` - global rotation, scale, and translation with
  optional 3D box support, operating on the ``coord`` key.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


def _build_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Build a 3D rotation matrix for one axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    if axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    raise NotImplementedError(f"Unsupported rotation axis: {axis}")


def _resolve_rotation_center(coord: np.ndarray, configured_center: np.ndarray | None) -> np.ndarray:
    """Resolve the rotation center for a point cloud."""
    if configured_center is not None:
        return configured_center
    return (coord.min(axis=0) + coord.max(axis=0)) / 2.0


class RandomRotateTargetAngle(BaseTransform):
    """Rotate point coordinates by one sampled discrete target angle."""

    _required_keys = ["coord"]

    def __init__(
        self,
        angle: Sequence[float],
        axis: str = "z",
        center: Sequence[float] | None = None,
        p: float = 0.5,
    ) -> None:
        """Initialize the target-angle rotation transform.

        Args:
            angle: Candidate rotation angles in multiples of ``pi`` radians.
            axis: Rotation axis. Only ``z`` is supported.
            center: Optional rotation center.
            p: Probability of applying the transform.
        """
        self.angle = list(angle)
        self.axis = axis
        self.center = np.asarray(center, dtype=np.float32) if center is not None else None
        self.p = p

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Rotate the sample by one selected target angle.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        angle = float(np.random.choice(self.angle)) * np.pi
        rotation = _build_rotation_matrix(self.axis, angle)
        coord = input_dict["coord"]
        center = _resolve_rotation_center(coord, self.center)
        input_dict["coord"] = (coord - center) @ rotation.T + center
        if "normal" in input_dict:
            input_dict["normal"] = input_dict["normal"] @ rotation.T
        return input_dict


class RandomFlip(BaseTransform):
    """Randomly flip point coordinates across BEV axes.

    Required keys:
        coord: Point coordinates (Nx3, float32).

    Optional keys:
        gt_boxes: 3D bounding boxes (Nx7 or Nx9). Flipped in-place when present.
        normal: Surface normals (Nx3). Flipped component-wise when present.

    Generated keys:
        coord: Updated coordinates.
        gt_boxes: Updated boxes (when present).
        normal: Updated normals (when present).
    """

    _required_keys = ["coord"]
    p = None

    def __init__(
        self,
        flip_ratio_bev_horizontal: float = 0.5,
        flip_ratio_bev_vertical: float = 0.5,
    ) -> None:
        """Initialize the random flip transform.

        Args:
            flip_ratio_bev_horizontal: Probability of flipping across the y-axis (horizontal).
            flip_ratio_bev_vertical: Probability of flipping across the x-axis (vertical).
        """
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Flip point coordinates across x and y axes.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if np.random.rand() < self.flip_ratio_bev_horizontal:
            input_dict["coord"][:, 1] *= -1.0
            if "normal" in input_dict:
                input_dict["normal"][:, 1] *= -1.0
            self._flip_boxes(input_dict, horizontal=True)

        if np.random.rand() < self.flip_ratio_bev_vertical:
            input_dict["coord"][:, 0] *= -1.0
            if "normal" in input_dict:
                input_dict["normal"][:, 0] *= -1.0
            self._flip_boxes(input_dict, horizontal=False)

        return input_dict

    def _flip_boxes(self, input_dict: dict[str, Any], horizontal: bool) -> None:
        """Flip 3D bounding boxes in-place when present."""
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
    """Apply global rotation, scaling, and optional translation to a point cloud.

    Operates on the ``coord`` key (Nx3 float32) and optionally updates
    ``gt_boxes``, ``normal``, and ``strength`` arrays.

    Required keys:
        coord: Point coordinates (Nx3, float32).

    Optional keys:
        gt_boxes: 3D bounding boxes updated consistently with the transform.
        normal: Surface normals rotated (but not scaled/translated).

    Generated keys:
        coord: Updated coordinates.
        gt_boxes: Updated boxes (when present).
        normal: Updated normals (when present).
    """

    _required_keys = ["coord"]

    def __init__(
        self,
        rot_range: list[float],
        scale_ratio_range: list[float],
        translation_std: list[float] | None = None,
    ) -> None:
        """Initialize the global rotation/scale/translation transform.

        Args:
            rot_range: Min and max rotation angles in radians (around the z-axis).
            scale_ratio_range: Min and max scale factors.
            translation_std: Standard deviation for Gaussian translation noise
                per axis ``[std_x, std_y, std_z]``. When ``None``, no
                translation is applied.
        """
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            np.asarray(translation_std, dtype=np.float32) if translation_std is not None else None
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply rotation, scale, and optional translation to coord and boxes.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        coord = input_dict["coord"].copy()

        rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)
        rotation_matrix = np.array(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])

        if self.translation_std is not None:
            translation = np.random.normal(0.0, self.translation_std, size=(1, 3)).astype(
                np.float32
            )
        else:
            translation = np.zeros((1, 3), dtype=np.float32)

        input_dict["coord"] = (coord @ rotation_matrix.T) * scale + translation

        if "normal" in input_dict:
            input_dict["normal"] = input_dict["normal"] @ rotation_matrix.T

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
        """Update 3D bounding boxes consistently with the point-cloud transform."""
        if "gt_boxes" not in input_dict:
            return

        boxes = input_dict["gt_boxes"].copy()
        boxes[:, :3] = (boxes[:, :3] @ rotation_matrix.T) * scale + translation
        boxes[:, 3:6] *= scale
        boxes[:, 6] += rotation
        if boxes.shape[1] >= 9:
            boxes[:, 7:9] = boxes[:, 7:9] @ rotation_matrix[:2, :2].T
        input_dict["gt_boxes"] = boxes
