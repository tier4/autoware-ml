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

"""Reconstruct where each point sat before ego-motion correction.

T4Dataset stores ego-motion-corrected points: every return is projected to where it would have
been had the whole sweep been captured instantaneously at the frame's reference time. The vehicle's
sensing pipeline evaluates the Nebula downsample mask and the ego crop box *before* that correction,
on the raw sweep. Reproducing those filters faithfully therefore means undoing the correction for
the duration of the keep/drop decision.

Measured end to end -- reconstructing a recorded concatenated cloud from an unfiltered T4Dataset --
taking both decisions in pre-correction space raises voxel IoU at 0.12 m from 0.702 to 0.878. The
plain point count barely moves (+0.08% to +0.32%), so count alone does not show this.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform

# Sample key holding the pre-correction (raw sweep) coordinates, in the ego frame.
PRE_CORRECTION_POINTS_KEY = "pre_correction_points"


@dataclass(frozen=True)
class PoseTable:
    """Time-ordered ego poses in the global frame."""

    times: npt.NDArray[np.float64]
    translations: npt.NDArray[np.float64]
    quaternions: npt.NDArray[np.float64]


def normalize_quaternions(quaternions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Scale each ``(w, x, y, z)`` row to unit length."""
    return quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


def quat_to_rotmat(quaternions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert ``(N, 4)`` ``(w, x, y, z)`` quaternions to ``(N, 3, 3)`` rotation matrices."""
    q = normalize_quaternions(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def slerp(
    q0: npt.NDArray[np.float64], q1: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation, falling back to lerp for nearly parallel rotations."""
    q0 = normalize_quaternions(q0)
    q1 = normalize_quaternions(q1)
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.abs(dot)
    close = dot > 0.9995
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1e-12)
    s1 = sin_theta / np.maximum(sin_theta_0, 1e-12)
    out = s0[:, None] * q0 + s1[:, None] * q1
    linear = q0 + alpha[:, None] * (q1 - q0)
    out = np.where(close[:, None], linear, out)
    return normalize_quaternions(out)


def load_pose_table(path: str | Path) -> PoseTable:
    """Read a T4 ``annotation/ego_pose.json`` into a time-ordered table."""
    poses = json.loads(Path(path).read_text())
    times = np.asarray([pose["timestamp"] * 1e-6 for pose in poses], dtype=np.float64)
    translations = np.asarray([pose["translation"] for pose in poses], dtype=np.float64)
    quaternions = np.asarray([pose["rotation"] for pose in poses], dtype=np.float64)
    order = np.argsort(times)
    return PoseTable(times[order], translations[order], normalize_quaternions(quaternions[order]))


def interpolate_poses(
    times: npt.NDArray[np.float64], pose_table: PoseTable
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interpolate ego translation and rotation at arbitrary times, clamping past the ends."""
    indices = np.searchsorted(pose_table.times, times, side="right")
    i0 = np.clip(indices - 1, 0, pose_table.times.shape[0] - 1)
    i1 = np.clip(indices, 0, pose_table.times.shape[0] - 1)
    t0 = pose_table.times[i0]
    t1 = pose_table.times[i1]
    alpha = np.clip((times - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    translations = (
        pose_table.translations[i0] * (1.0 - alpha[:, None])
        + pose_table.translations[i1] * alpha[:, None]
    )
    quaternions = slerp(pose_table.quaternions[i0], pose_table.quaternions[i1], alpha)
    return translations, quaternions


def invert_ego_motion(
    corrected_ego: npt.NDArray[np.float32],
    reference_time: float,
    point_times: npt.NDArray[np.float64],
    pose_table: PoseTable,
) -> npt.NDArray[np.float64]:
    """Map corrected ego-frame points back to the ego frame at their own acquisition time.

    Args:
        corrected_ego: ``(N, 3)`` points in the ego frame at ``reference_time``.
        reference_time: Frame reference time in seconds.
        point_times: ``(N,)`` per-point acquisition times in seconds.
        pose_table: Ego poses covering the sweep.

    Returns:
        ``(N, 3)`` points in the ego frame as it stood when each point was captured.
    """
    ref_t, ref_q = interpolate_poses(np.array([reference_time], dtype=np.float64), pose_table)
    global_points = corrected_ego.astype(np.float64) @ quat_to_rotmat(ref_q)[0].T + ref_t[0]
    point_t, point_q = interpolate_poses(point_times, pose_table)
    return np.einsum("nj,njk->nk", global_points - point_t, quat_to_rotmat(point_q))


def point_offsets_seconds(
    timestamps: npt.NDArray[np.float32], unit: str
) -> npt.NDArray[np.float64]:
    """Convert the per-point timestamp feature to seconds."""
    scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported timestamp unit: {unit}")
    return timestamps.astype(np.float64) * scale


def stamp_to_seconds(stamp: Mapping[str, int]) -> float:
    """Convert a ``{sec, nanosec}`` stamp to seconds."""
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


class InvertEgoMotionCorrection(BaseTransform):
    """Attach pre-correction coordinates for the vehicle-side filters to decide on.

    Writes a ``(N, 3)`` array under :data:`PRE_CORRECTION_POINTS_KEY`, holding each point's position
    in the ego frame *at its own acquisition time* rather than at the frame reference time. Run this
    before :class:`~autoware_ml.transforms.point_cloud.nebula_mask.NebulaDownsampleMaskFilter` and
    :class:`~autoware_ml.transforms.point_cloud.crop_box.CropBoxFilter`; both prefer these
    coordinates when present, because the vehicle evaluates them pre-correction.

    The attached array is per-point and the same length as ``points``, so the filters' aligned-array
    handling keeps the two in lockstep as points are dropped. Both this transform and its consumers
    assert that invariant rather than trusting it.

    ``points`` themselves are never modified: only the keep/drop decision moves into pre-correction
    space, and surviving points are emitted with their original corrected coordinates.
    """

    _required_keys = ["points"]

    def __init__(
        self,
        *,
        timestamp_dim: int = 6,
        timestamp_unit: str = "ns",
        ego_pose_path: str | None = None,
        strict: bool = True,
    ) -> None:
        """Initialize the InvertEgoMotionCorrection transform.

        Args:
            timestamp_dim: Index of the per-point feature holding the acquisition-time offset.
            timestamp_unit: Unit of that feature: ``"ns"``, ``"us"``, ``"ms"`` or ``"s"``.
            ego_pose_path: Explicit path to ``ego_pose.json``. When omitted it is resolved from the
                sample's ``lidar_path`` by walking up to the scene directory.
            strict: Raise when the sample lacks the metadata needed to invert the correction. Set
                ``False`` to pass such samples through untouched, leaving the filters to fall back
                to corrected coordinates.
        """
        self.timestamp_dim = int(timestamp_dim)
        self.timestamp_unit = str(timestamp_unit)
        self.ego_pose_path = ego_pose_path
        self.strict = bool(strict)
        self._pose_tables: dict[str, PoseTable] = {}

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Attach pre-correction coordinates, or pass through when metadata is unavailable."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        sources_info = input_dict.get("lidar_sources_info")
        if not isinstance(sources_info, Mapping) or points.shape[1] <= self.timestamp_dim:
            if self.strict:
                raise KeyError(
                    "InvertEgoMotionCorrection needs 'lidar_sources_info' and a per-point "
                    f"timestamp at dim {self.timestamp_dim}; got "
                    f"{points.shape[1]} point features."
                )
            return input_dict

        pose_table = self._pose_table(input_dict)
        if pose_table is None:
            return input_dict

        reference_time = stamp_to_seconds(sources_info["stamp"])
        # Default to the corrected coordinates so sources absent from the metadata still get a
        # sane value rather than zeros, which would sit inside the ego crop box.
        pre_correction = points[:, :3].astype(np.float64, copy=True)
        for source in sources_info.get("sources", ()):
            begin = int(source["idx_begin"])
            end = begin + int(source["length"])
            if end > points.shape[0]:
                raise ValueError(
                    f"lidar_sources_info source range {begin}:{end} exceeds "
                    f"{points.shape[0]} points."
                )
            if end <= begin:
                continue
            offsets = point_offsets_seconds(
                points[begin:end, self.timestamp_dim], self.timestamp_unit
            )
            point_times = stamp_to_seconds(source["stamp"]) + offsets
            pre_correction[begin:end] = invert_ego_motion(
                points[begin:end, :3], reference_time, point_times, pose_table
            )

        result = pre_correction.astype(np.float32, copy=False)
        _assert_aligned(result, points.shape[0])
        input_dict[PRE_CORRECTION_POINTS_KEY] = result
        return input_dict

    def _pose_table(self, input_dict: Mapping[str, Any]) -> PoseTable | None:
        path = self.ego_pose_path or _resolve_ego_pose_path(input_dict.get("lidar_path"))
        if path is None:
            if self.strict:
                raise KeyError(
                    "InvertEgoMotionCorrection could not locate ego_pose.json; pass "
                    "ego_pose_path explicitly or provide 'lidar_path'."
                )
            return None
        key = str(path)
        if key not in self._pose_tables:
            self._pose_tables[key] = load_pose_table(path)
        return self._pose_tables[key]


def _resolve_ego_pose_path(lidar_path: Any) -> Path | None:
    """Find ``annotation/ego_pose.json`` for the scene owning ``lidar_path``."""
    if not lidar_path:
        return None
    for parent in Path(lidar_path).resolve().parents:
        candidate = parent / "annotation" / "ego_pose.json"
        if candidate.is_file():
            return candidate
    return None


def _assert_aligned(values: npt.NDArray[Any], point_count: int) -> None:
    if values.shape[0] != point_count:
        raise ValueError(
            f"{PRE_CORRECTION_POINTS_KEY} holds {values.shape[0]} rows but there are "
            f"{point_count} points; the two must stay aligned."
        )


def pre_correction_points(
    input_dict: Mapping[str, Any], points: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32] | None:
    """Return the sample's pre-correction coordinates, checking they align with ``points``.

    Returns ``None`` when the sample carries none, so callers fall back to corrected coordinates.
    """
    values = input_dict.get(PRE_CORRECTION_POINTS_KEY)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    _assert_aligned(values, points.shape[0])
    return values
