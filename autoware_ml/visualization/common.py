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

"""Shared conversion helpers for visualization adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from autoware_ml.visualization.colors import scalar_to_heatmap_colors
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    AnnotationInfo,
    PointCloud3DEvent,
    TextEvent,
    VisualizationEvent,
)

POINT_COLOR_MODES = frozenset({"semantic", "intensity", "solid"})


def as_numpy(data: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert tensors and sequences into NumPy arrays."""
    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        array = np.asarray(data)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def ensure_xyz(points: Any) -> np.ndarray:
    """Extract xyz coordinates from a point tensor or array."""
    array = as_numpy(points, np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {array.shape}")
    return array[:, :3]


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


def build_scene_extent_event(
    points: Any | None,
    boxes: Sequence[Any | None] = (),
    *,
    root_path: str = "scene",
) -> PointCloud3DEvent | None:
    """Build two invisible corner points spanning everything one frame draws.

    Rerun fits the initial eye of a 3D view to the bounds of the entities that
    view shows. Sibling views of the same frame therefore start from different
    eyes as soon as one of them shows boxes that stick out of the point cloud.
    Logging the union extent once, and listing it in every scene view, gives
    those views identical bounds and so an identical starting eye.
    """
    corners: list[np.ndarray] = []
    if points is not None:
        xyz = ensure_xyz(points)
        if xyz.shape[0]:
            corners.append(xyz)
    for box_array in boxes:
        if box_array is None:
            continue
        array = as_numpy(box_array, np.float32)
        if array.ndim != 2 or array.shape[1] < 6 or not array.shape[0]:
            continue
        # Half the box diagonal bounds every yaw, so no rotation is needed.
        half_diagonal = np.linalg.norm(array[:, 3:6], axis=1, keepdims=True) * 0.5
        corners.append(array[:, :3] - half_diagonal)
        corners.append(array[:, :3] + half_diagonal)
    if not corners:
        return None
    stacked = np.concatenate(corners, axis=0)
    positions = np.stack([stacked.min(axis=0), stacked.max(axis=0)]).astype(np.float32)
    return PointCloud3DEvent(
        path=f"{root_path}/meta/extent",
        positions=positions,
        colors=np.zeros((2, 4), dtype=np.uint8),
        radii=1e-3,
    )


def ensure_image_uint8(image: Any) -> np.ndarray:
    """Normalize image-like data to an HWC uint8 array."""
    array = as_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"image must have shape (H, W, C), got {array.shape}")
    if array.shape[0] <= 8 and array.shape[1] > 8 and array.shape[2] > 8:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    return array


def format_class_label(
    class_id: int,
    class_names: Sequence[str] | None = None,
    score: float | None = None,
) -> str:
    """Build a readable class label string."""
    if class_names is not None and 0 <= class_id < len(class_names):
        base = class_names[class_id]
    else:
        base = str(class_id)
    if score is None:
        return base
    return f"{base} ({score:.2f})"


def resolve_palette_size(
    label_arrays: Sequence[np.ndarray | None],
    class_names: Sequence[str] | None = None,
) -> int:
    """Return a palette size covering every declared class and observed label.

    Sizing on the declared ``class_names`` as well as the labels present in the
    current sample keeps the viewer legend stable while scrubbing the timeline:
    without it, a sample containing only two of ten classes would publish a
    two-entry legend and leave the remaining classes unnamed.
    """
    palette_size = 0
    for labels in label_arrays:
        if labels is None or labels.size == 0:
            continue
        valid_labels = labels[labels >= 0]
        if valid_labels.size > 0:
            palette_size = max(palette_size, int(valid_labels.max()) + 1)
    if class_names is not None:
        palette_size = max(palette_size, len(class_names))
    return palette_size


def build_class_annotation_context(
    root_path: str,
    palette: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> AnnotationContextEvent | None:
    """Build one semantic legend for visualization descendants."""
    if palette.shape[0] == 0:
        return None
    return AnnotationContextEvent(
        path=root_path,
        annotations=[
            AnnotationInfo(
                id=index,
                label=format_class_label(index, class_names),
                color=tuple(int(channel) for channel in palette[index]),
            )
            for index in range(palette.shape[0])
        ],
    )


def build_sample_metadata_events(
    root_path: str, sample_name: str | None
) -> list[VisualizationEvent]:
    """Build lightweight per-sample metadata events."""
    if sample_name is None:
        return []
    return [TextEvent(f"{root_path}/meta/sample", sample_name)]
