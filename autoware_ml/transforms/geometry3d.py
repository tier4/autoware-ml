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

"""Shared geometric operations for 3D scene augmentations.

This module holds the *math* behind the rotation / scale / translation / flip
augmentations as plain functions. It owns no transform classes and declares no
required keys, so the modality-specific transforms in
``transforms.point_cloud.geometry``, ``transforms.camera_lidar.geometry`` and
``transforms.camera.geometry`` can all reuse exactly the same computations
(verified to be identical by the cross-namespace tests).

Two groups of helpers:

* **pure array math** - matrices and per-array transforms that take and return
  numpy arrays (``rotation_matrix``, ``rot_scale_trans_matrix``,
  ``flip_matrix``);
* **sampling + dict application** - draw augmentation parameters and apply them
  in place to whichever of ``coord`` / ``points`` / ``normal`` / ``gt_boxes`` /
  camera matrices a transform decides to touch. The transform classes choose
  *which* helpers to call (and require the matching keys); these helpers never
  silently skip work based on key presence beyond the documented point/normal
  optionality.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

POINT_KEYS = ("coord", "points")


def rotation_matrix(axis: str, angle: float) -> npt.NDArray[np.float32]:
    """Build a 3x3 rotation matrix for a single axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    if axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    raise NotImplementedError(f"Unsupported rotation axis: {axis}")


def resolve_rotation_center(
    coord: npt.NDArray[np.float32], configured_center: npt.NDArray[np.float32] | None
) -> npt.NDArray[np.float32]:
    """Resolve the rotation center for a point cloud (config value or bbox center)."""
    if configured_center is not None:
        return configured_center
    return (coord.min(axis=0) + coord.max(axis=0)) / 2.0


def rot_scale_trans_matrix(
    rotation: npt.NDArray[np.float32], scale: float, translation: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Compose a 4x4 point-space augmentation from rotation, scale, translation."""
    augmentation = np.eye(4, dtype=np.float32)
    augmentation[:3, :3] = rotation * scale
    augmentation[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return augmentation


def flip_matrix(flip_x: bool, flip_y: bool) -> npt.NDArray[np.float32]:
    """Compose a 4x4 flip matrix (negate x and/or y)."""
    flip = np.eye(4, dtype=np.float32)
    if flip_x:
        flip[0, 0] = -1.0
    if flip_y:
        flip[1, 1] = -1.0
    return flip


def sample_rot_scale_trans(
    rot_range: Sequence[float],
    scale_ratio_range: Sequence[float],
    translation_std: npt.NDArray[np.float32] | None,
) -> tuple[npt.NDArray[np.float32], float, float, npt.NDArray[np.float32]]:
    """Sample a z-rotation, scale, and translation for a global scene transform.

    Returns ``(rotation_matrix, rotation_angle, scale, translation)`` where
    ``translation`` has shape ``(1, 3)`` (zeros when ``translation_std`` is None).
    """
    rotation = float(np.random.uniform(rot_range[0], rot_range[1]))
    matrix = rotation_matrix("z", rotation)
    scale = float(np.random.uniform(scale_ratio_range[0], scale_ratio_range[1]))
    if translation_std is not None:
        translation = np.random.normal(0.0, translation_std, size=(1, 3)).astype(np.float32)
    else:
        translation = np.zeros((1, 3), dtype=np.float32)
    return matrix, rotation, scale, translation


def sample_bev_flips(
    flip_ratio_bev_horizontal: float, flip_ratio_bev_vertical: float
) -> tuple[bool, bool]:
    """Sample BEV flips. Returns ``(flip_x, flip_y)`` (longitudinal, lateral)."""
    flip_y = bool(np.random.rand() < flip_ratio_bev_horizontal)
    flip_x = bool(np.random.rand() < flip_ratio_bev_vertical)
    return flip_x, flip_y


def has_point_cloud(input_dict: dict[str, Any]) -> bool:
    """Return whether any point representation (``coord`` / ``points``) is present."""
    return any(input_dict.get(key) is not None for key in POINT_KEYS)


def require_point_cloud(input_dict: dict[str, Any]) -> None:
    """Raise if no point representation is present (no silent skip)."""
    if not has_point_cloud(input_dict):
        raise KeyError(
            f"a point representation ({' or '.join(POINT_KEYS)}) is required but none was found"
        )


def apply_to_point_xyz(
    input_dict: dict[str, Any], fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
) -> None:
    """Apply ``fn`` to the XYZ columns of every present point representation."""
    for key in POINT_KEYS:
        array = input_dict.get(key)
        if array is None:
            continue
        array = np.asarray(array).copy()
        array[:, :3] = fn(array[:, :3]).astype(array.dtype)
        input_dict[key] = array


def transform_points(
    input_dict: dict[str, Any],
    rotation: npt.NDArray[np.float32],
    scale: float,
    translation: npt.NDArray[np.float32],
) -> None:
    """Rotate, scale, and translate every present point representation."""
    apply_to_point_xyz(input_dict, lambda xyz: (xyz @ rotation.T) * scale + translation)


def rotate_points_about_center(
    input_dict: dict[str, Any], rotation: npt.NDArray[np.float32], center: npt.NDArray[np.float32]
) -> None:
    """Rotate every present point representation about ``center``."""
    apply_to_point_xyz(input_dict, lambda xyz: (xyz - center) @ rotation.T + center)


def flip_points(input_dict: dict[str, Any], axis: int) -> None:
    """Negate one axis of every present point representation."""

    def negate(xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        xyz = xyz.copy()
        xyz[:, axis] *= -1.0
        return xyz

    apply_to_point_xyz(input_dict, negate)


def transform_normal(input_dict: dict[str, Any], rotation: npt.NDArray[np.float32]) -> None:
    """Rotate per-point ``normal`` vectors when present."""
    if "normal" in input_dict:
        input_dict["normal"] = np.asarray(input_dict["normal"]) @ rotation.T


def flip_normal(input_dict: dict[str, Any], axis: int) -> None:
    """Negate one axis of per-point ``normal`` vectors when present."""
    if "normal" in input_dict:
        normal = np.asarray(input_dict["normal"]).copy()
        normal[:, axis] *= -1.0
        input_dict["normal"] = normal


def transform_boxes(
    input_dict: dict[str, Any],
    rotation: npt.NDArray[np.float32],
    rotation_angle: float,
    scale: float,
    translation: npt.NDArray[np.float32],
) -> None:
    """Update ``gt_boxes`` consistently with a global rotation/scale/translation."""
    if "gt_boxes" not in input_dict:
        return
    boxes = np.asarray(input_dict["gt_boxes"]).copy()
    boxes[:, :3] = (boxes[:, :3] @ rotation.T) * scale + translation
    boxes[:, 3:6] *= scale
    if boxes.shape[1] > 6:
        boxes[:, 6] += rotation_angle
    if boxes.shape[1] >= 9:
        boxes[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    input_dict["gt_boxes"] = boxes


def flip_boxes(input_dict: dict[str, Any], axis: int) -> None:
    """Flip ``gt_boxes`` across one BEV axis (``axis=1`` lateral, ``axis=0`` longitudinal)."""
    if "gt_boxes" not in input_dict:
        return
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 (x / longitudinal) or 1 (y / lateral), got {axis}")
    boxes = np.asarray(input_dict["gt_boxes"]).copy()
    if axis == 1:  # lateral flip: negate y, mirror yaw and y-velocity
        boxes[:, 1] *= -1.0
        if boxes.shape[1] > 6:
            boxes[:, 6] *= -1.0
        if boxes.shape[1] >= 9:
            boxes[:, 8] *= -1.0
    else:  # longitudinal flip: negate x, reflect yaw and x-velocity
        boxes[:, 0] *= -1.0
        if boxes.shape[1] > 6:
            boxes[:, 6] = np.pi - boxes[:, 6]
        if boxes.shape[1] >= 9:
            boxes[:, 7] *= -1.0
    input_dict["gt_boxes"] = boxes


def update_camera_matrices(input_dict: dict[str, Any], aug_inv: npt.NDArray[np.float32]) -> None:
    """Keep camera projection consistent after a lidar-space transform.

    Applies ``aug_inv`` (inverse of the 4x4 point-space augmentation) to
    ``lidar2cam`` and recomputes ``lidar2img`` from ``camera_intrinsics`` when
    available, otherwise applies ``aug_inv`` to ``lidar2img`` directly.
    """
    lidar2cam = np.asarray(input_dict["lidar2cam"], dtype=np.float32) @ aug_inv
    input_dict["lidar2cam"] = lidar2cam
    if "camera_intrinsics" in input_dict:
        input_dict["lidar2img"] = (
            np.asarray(input_dict["camera_intrinsics"], dtype=np.float32) @ lidar2cam
        )
    elif "lidar2img" in input_dict:
        input_dict["lidar2img"] = np.asarray(input_dict["lidar2img"], dtype=np.float32) @ aug_inv
