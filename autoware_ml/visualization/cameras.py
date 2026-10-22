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

"""Camera visualization adapters.

Logs each camera as a Transform3D + Pinhole + Image triplet so that Rerun
can project 3D points onto camera images on hover.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.visualization.colors import (
    build_label_palette,
    depths_to_colors,
    labels_to_colors,
)
from autoware_ml.visualization.common import (
    as_numpy,
    build_class_annotation_context,
    ensure_xyz,
    resolve_palette_size,
)
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    ImageEvent,
    LineStrips2DEvent,
    PinholeEvent,
    Points2DEvent,
    Transform3DEvent,
    VisualizationEvent,
)

_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


@dataclass(frozen=True)
class CameraPointProjection:
    """Describe one LiDAR point layer projected into every camera image."""

    points: Any
    labels: Any | None = None
    class_names: Sequence[str] | None = None
    ignore_index: int | None = None
    colors: Any | None = None
    radius: float = 1.5


@dataclass(frozen=True)
class CameraBoxProjection:
    """Describe one 3D box layer projected into every camera image."""

    boxes: Any
    labels: Any
    class_names: Sequence[str] | None = None
    radius: float = 1.5


def build_camera_events(
    images: dict[str, Any],
    *,
    root_path: str = "scene/cameras",
    point_layers: Mapping[str, CameraPointProjection] | None = None,
    box_layers: Mapping[str, CameraBoxProjection] | None = None,
    max_projected_points: int = 10_000,
) -> list[VisualizationEvent]:
    """Build visualization events for all cameras in one sample.

    Each camera gets a ``Transform3D`` (lidar-to-camera extrinsic expressed as
    child-from-parent), a ``Pinhole`` (intrinsic), and an ``Image`` logged
    under ``{root_path}/{cam_name}``.  Rerun uses the transform hierarchy to
    enable automatic point-to-image projection when hovering over 3D points.

    Args:
        images: Per-camera dict from a T4Dataset batch entry.  Each value must
            contain ``img_path`` (str), ``cam2img`` (3×3 array-like), and
            ``lidar2cam`` (4×4 array-like).
        root_path: Entity path prefix for all camera entities.
        point_layers: Named LiDAR layers to project persistently onto each image.
        box_layers: Named 3D box layers to render as projected wireframes.
        max_projected_points: Maximum points retained per layer and camera.

    Returns:
        List of ``Transform3DEvent``, ``PinholeEvent``, and ``ImageEvent``
        objects, one triplet per camera.

    Raises:
        ValueError: If a camera entry is missing calibration keys.
        FileNotFoundError: If a camera image cannot be read from disk.
    """
    if max_projected_points <= 0:
        raise ValueError("max_projected_points must be greater than zero")

    events: list[VisualizationEvent] = []
    for cam_name, cam_info in images.items():
        cam_path = f"{root_path}/{cam_name}"
        camera_events, intrinsic, extrinsic, resolution = _build_single_camera_events(
            cam_info, cam_path, cam_name
        )
        events.extend(camera_events)
        for layer_name, layer in (point_layers or {}).items():
            events.extend(
                _build_projected_point_events(
                    layer,
                    path=f"{cam_path}/projected/{layer_name}",
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    resolution=resolution,
                    max_points=max_projected_points,
                )
            )
        for layer_name, layer in (box_layers or {}).items():
            events.extend(
                _build_projected_box_events(
                    layer,
                    path=f"{cam_path}/projected/{layer_name}",
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    resolution=resolution,
                )
            )
    return events


def _build_single_camera_events(
    cam_info: dict[str, Any],
    cam_path: str,
    cam_name: str,
) -> tuple[list[VisualizationEvent], np.ndarray, np.ndarray, tuple[int, int]]:
    """Build the three visualization events for one camera."""
    missing = [key for key in ("img_path", "cam2img", "lidar2cam") if cam_info.get(key) is None]
    if missing:
        raise ValueError(
            f"Camera {cam_name!r} is missing required calibration keys: {', '.join(missing)}."
        )

    img_path = cam_info["img_path"]
    cam2img = cam_info["cam2img"]
    lidar2cam = cam_info["lidar2cam"]

    image_array = _load_image(str(img_path))
    height, width = image_array.shape[:2]

    intrinsic = np.asarray(cam2img, dtype=np.float32).reshape(3, 3)
    extrinsic = np.asarray(lidar2cam, dtype=np.float32).reshape(4, 4)

    resolution = (width, height)
    return (
        [
            Transform3DEvent(
                path=cam_path,
                translation=extrinsic[:3, 3],
                rotation_matrix=extrinsic[:3, :3],
            ),
            PinholeEvent(
                path=cam_path,
                image_from_camera=intrinsic,
                resolution=resolution,
            ),
            ImageEvent(
                path=cam_path,
                image=image_array,
            ),
        ],
        intrinsic,
        extrinsic,
        resolution,
    )


def _project_points(
    points: Any,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    resolution: tuple[int, int],
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points and return pixels, source indices, and depths."""
    positions = ensure_xyz(points)
    homogeneous = np.concatenate(
        (positions, np.ones((positions.shape[0], 1), dtype=np.float32)), axis=1
    )
    camera_points = (extrinsic @ homogeneous.T).T[:, :3]
    width, height = resolution
    valid = camera_points[:, 2] > 1e-5
    source_indices = np.flatnonzero(valid)
    camera_points = camera_points[valid]
    if camera_points.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    projected = camera_points @ intrinsic.T
    projected = projected[:, :2] / projected[:, 2:3]
    in_frame = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    projected = projected[in_frame].astype(np.float32, copy=False)
    source_indices = source_indices[in_frame]
    depths = camera_points[in_frame, 2].astype(np.float32, copy=False)
    if projected.shape[0] > max_points:
        keep = np.linspace(0, projected.shape[0] - 1, max_points, dtype=np.int64)
        projected = projected[keep]
        source_indices = source_indices[keep]
        depths = depths[keep]
    return projected, source_indices, depths


def _build_projected_point_events(
    layer: CameraPointProjection,
    *,
    path: str,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    resolution: tuple[int, int],
    max_points: int,
) -> list[VisualizationEvent]:
    """Build one persistent point overlay for a camera image."""
    positions = ensure_xyz(layer.points)
    labels = as_numpy(layer.labels, np.int64).reshape(-1) if layer.labels is not None else None
    if labels is not None and labels.shape[0] != positions.shape[0]:
        raise ValueError(f"Camera point layer {path!r} labels must align with points")
    colors = as_numpy(layer.colors, np.uint8) if layer.colors is not None else None
    if colors is not None and (colors.ndim != 2 or colors.shape[0] != positions.shape[0]):
        raise ValueError(f"Camera point layer {path!r} colors must align with points")

    pixels, source_indices, depths = _project_points(
        positions,
        intrinsic,
        extrinsic,
        resolution,
        max_points=max_points,
    )
    if pixels.shape[0] == 0:
        return []

    events: list[VisualizationEvent] = []
    projected_labels = labels[source_indices] if labels is not None else None
    if projected_labels is not None:
        palette = build_label_palette(resolve_palette_size([projected_labels], layer.class_names))
        annotation_context = build_class_annotation_context(path, palette, layer.class_names)
        if annotation_context is not None:
            events.append(annotation_context)
        projected_colors = labels_to_colors(
            projected_labels,
            palette,
            ignore_index=layer.ignore_index,
        )
    elif colors is not None:
        projected_colors = colors[source_indices]
    else:
        projected_colors = depths_to_colors(depths)

    events.append(
        Points2DEvent(
            path=path,
            positions=pixels,
            colors=projected_colors,
            radii=float(layer.radius),
            class_ids=projected_labels,
        )
    )
    return events


def _box_corners(box: np.ndarray) -> np.ndarray:
    """Return eight corners for one yaw-only box in LiDAR coordinates."""
    half = box[3:6] * 0.5
    corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    cosine, sine = float(np.cos(box[6])), float(np.sin(box[6]))
    corners[:, :2] = corners[:, :2] @ np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    return corners + box[:3]


def _project_box_edges(
    boxes: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Project visible edges and return their source box indices."""
    width, height = resolution
    strips: list[np.ndarray] = []
    box_indices: list[int] = []
    for box_index, box in enumerate(boxes):
        corners = _box_corners(box)
        homogeneous = np.concatenate((corners, np.ones((8, 1), dtype=np.float32)), axis=1)
        camera_corners = (extrinsic @ homogeneous.T).T[:, :3]
        valid_edges = [edge for edge in _BOX_EDGES if np.all(camera_corners[list(edge), 2] > 1e-5)]
        if not valid_edges:
            continue
        projected = camera_corners @ intrinsic.T
        projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-5, None)
        visible_points = projected[np.unique(np.asarray(valid_edges, dtype=np.int64))]
        if (
            visible_points[:, 0].max() < 0.0
            or visible_points[:, 0].min() >= width
            or visible_points[:, 1].max() < 0.0
            or visible_points[:, 1].min() >= height
        ):
            continue
        for start, end in valid_edges:
            strips.append(projected[[start, end]].astype(np.float32, copy=False))
            box_indices.append(box_index)
    return strips, np.asarray(box_indices, dtype=np.int64)


def _build_projected_box_events(
    layer: CameraBoxProjection,
    *,
    path: str,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    resolution: tuple[int, int],
) -> list[VisualizationEvent]:
    """Build class-colored projected 3D box wireframes for one camera."""
    boxes = as_numpy(layer.boxes, np.float32)
    labels = as_numpy(layer.labels, np.int64).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"Camera box layer {path!r} boxes must have shape (N, >=7)")
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError(f"Camera box layer {path!r} labels must align with boxes")
    strips, box_indices = _project_box_edges(boxes, intrinsic, extrinsic, resolution)
    if not strips:
        return []

    palette = build_label_palette(resolve_palette_size([labels], layer.class_names))
    annotation_context: AnnotationContextEvent | None = build_class_annotation_context(
        path, palette, layer.class_names
    )
    projected_labels = labels[box_indices]
    events: list[VisualizationEvent] = []
    if annotation_context is not None:
        events.append(annotation_context)
    events.append(
        LineStrips2DEvent(
            path=path,
            strips=strips,
            colors=labels_to_colors(projected_labels, palette),
            radii=float(layer.radius),
            class_ids=projected_labels,
        )
    )
    return events


def _load_image(img_path: str) -> npt.NDArray[np.uint8]:
    """Load an RGB image from disk using cv2."""
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
