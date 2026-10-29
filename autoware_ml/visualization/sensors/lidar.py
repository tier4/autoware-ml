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

from typing import Any

import numpy as np

from autoware_ml.visualization.colors import scalar_to_heatmap_colors
from autoware_ml.visualization.common import as_numpy, ensure_xyz
from autoware_ml.visualization.viewers.events import PointCloud3DEvent

POINT_COLOR_MODES = frozenset({"semantic", "intensity", "solid"})


def point_cloud_colors(points: Any, mode: str = "semantic") -> np.ndarray | None:
    """Build optional RGBA point colors, including per-frame LiDAR intensity."""
    if mode not in POINT_COLOR_MODES:
        raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")
    if mode == "semantic":
        return None
    array = as_numpy(points, np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {array.shape}")
    if mode == "solid":
        colors = np.empty((array.shape[0], 4), dtype=np.uint8)
        colors[:] = (220, 230, 240, 255)
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
    points: Any,
    *,
    root_path: str,
    point_radius: float,
) -> list[PointCloud3DEvent]:
    """Build solid and, when available, intensity LiDAR scene layers.

    Both layers are logged so the viewer can switch coloring modes without
    rerunning inference. Semantic layers remain task-specific because they
    carry class ids and annotation contexts.
    """
    positions = ensure_xyz(points)
    # Rerun broadcasts a scalar radius to every point. Keeping this scalar
    # avoids serializing an identical float once per point and per color mode.
    radius = float(point_radius)
    events = [
        PointCloud3DEvent(
            path=f"{root_path}/lidar/solid",
            positions=positions,
            colors=point_cloud_colors(points, "solid"),
            radii=radius,
        )
    ]
    points_array = as_numpy(points)
    if points_array.ndim == 2 and points_array.shape[1] >= 4:
        events.append(
            PointCloud3DEvent(
                path=f"{root_path}/lidar/intensity",
                positions=positions,
                colors=point_cloud_colors(points, "intensity"),
                radii=radius,
            )
        )
    return events
