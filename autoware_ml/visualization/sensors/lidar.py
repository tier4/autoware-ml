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

"""LiDAR point-cloud layers shared by the task adapters."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32, UInt8
import numpy as np

from autoware_ml.visualization.colors import scalar_to_heatmap_colors
from autoware_ml.visualization.common import ensure_xyz
from autoware_ml.visualization.viewers.events import PointCloud3DEvent

POINT_COLOR_MODES = frozenset({"semantic", "intensity", "solid"})
# Neutral geometry color of the solid point layer.
SOLID_POINT_COLOR = (220, 230, 240, 255)


def point_cloud_colors(
    points: Float32[np.ndarray, "num_points num_features"], mode: str = "semantic"
) -> UInt8[np.ndarray, "num_points 4"] | None:
    """
    Build the RGBA point colors of one color mode, with the intensity normalized per frame.

    Args:
        points: Points with x, y, z and, for the intensity mode, the intensity column.
        mode: ``semantic``, ``intensity`` or ``solid``.

    Returns:
        UInt8[np.ndarray, "num_points 4"] | None: The colors, None for the semantic mode,
            whose colors come from the class of every point.

    Raises:
        ValueError: If the mode is unknown, or the points lack the columns it needs.
    """
    if mode not in POINT_COLOR_MODES:
        raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")
    if mode == "semantic":
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {array.shape}")
    if mode == "solid":
        colors = np.empty((array.shape[0], 4), dtype=np.uint8)
        colors[:] = SOLID_POINT_COLOR
        return colors
    if array.shape[1] < 4:
        raise ValueError("intensity coloring requires points with an intensity column")
    intensity = np.nan_to_num(array[:, 3], nan=0.0, posinf=0.0, neginf=0.0)
    low, high = (float(intensity.min()), float(intensity.max())) if intensity.size else (0.0, 0.0)
    normalized = np.zeros_like(intensity, dtype=np.float32)
    if high > low:
        normalized = (intensity - low) / (high - low)
    return scalar_to_heatmap_colors(normalized)


def build_lidar_reference_events(
    points: Float32[np.ndarray, "num_points num_features"],
    *,
    root_path: str,
    point_radius: float,
) -> Sequence[PointCloud3DEvent]:
    """
    Build the solid and, when available, the intensity LiDAR scene layers.

    Both layers are logged so the viewer can switch coloring modes without rerunning inference.
    Semantic layers remain task-specific because they carry class ids and annotation contexts.

    Args:
        points: Points of the frame.
        root_path: Entity path of the scene.
        point_radius: Radius of every point.

    Returns:
        Sequence[PointCloud3DEvent]: The solid layer, then the intensity layer when the points
            carry an intensity column.
    """
    positions = ensure_xyz(points)
    # Rerun broadcasts a scalar radius to every point. Keeping this scalar avoids serializing
    # an identical float once per point and per color mode.
    radius = float(point_radius)
    events = [
        PointCloud3DEvent(
            path=f"{root_path}/lidar/solid",
            positions=positions,
            colors=point_cloud_colors(points, "solid"),
            radii=radius,
        )
    ]
    if points.ndim == 2 and points.shape[1] >= 4:
        events.append(
            PointCloud3DEvent(
                path=f"{root_path}/lidar/intensity",
                positions=positions,
                colors=point_cloud_colors(points, "intensity"),
                radii=radius,
            )
        )
    return events
