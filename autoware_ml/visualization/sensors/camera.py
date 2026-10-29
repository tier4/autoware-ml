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
from pathlib import Path

import cv2
from jaxtyping import Float32, Int64, UInt8
import numpy as np

from autoware_ml.visualization.colors import (
    build_label_palette,
    depths_to_colors,
    labels_to_colors,
)
from autoware_ml.visualization.common import (
    build_class_annotation_context,
    ensure_xyz,
    resolve_palette_size,
)
from autoware_ml.visualization.entity_paths import CAMERAS_PATH
from autoware_ml.visualization.viewers.events import (
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
class CameraCalibration:
    """
    Image file and calibration of one camera of a sample.

    Attributes:
        image_path: Path of the camera image.
        intrinsics: Pinhole intrinsics of the image.
        lidar_to_camera: Homogeneous transform from the LiDAR frame to the camera frame.
    """

    image_path: str
    intrinsics: Float32[np.ndarray, "3 3"]
    lidar_to_camera: Float32[np.ndarray, "4 4"]


@dataclass(frozen=True)
class CameraPointProjection:
    """
    One LiDAR point layer projected into every camera image.

    Attributes:
        points: Points of the layer.
        labels: Class of every point, None to color the points by depth or by ``colors``.
        class_names: Class names by index, None to show the indices.
        ignore_index: Label drawn as unlabelled.
        colors: RGBA color of every point, used when the points carry no labels.
        radius: Radius of every projected point in pixels.
    """

    points: Float32[np.ndarray, "num_points num_features"]
    labels: Int64[np.ndarray, " num_points"] | None = None
    class_names: Sequence[str] | None = None
    ignore_index: int | None = None
    colors: UInt8[np.ndarray, "num_points 4"] | None = None
    radius: float = 1.5


@dataclass(frozen=True)
class CameraBoxProjection:
    """
    One 3D box layer projected into every camera image as wireframes.

    Attributes:
        boxes: Boxes, centre, size and yaw in the first seven columns.
        labels: Class of every box.
        class_names: Class names by index, None to show the indices.
        radius: Line radius of the wireframes in pixels.
    """

    boxes: Float32[np.ndarray, "num_boxes box_params"]
    labels: Int64[np.ndarray, " num_boxes"]
    class_names: Sequence[str] | None = None
    radius: float = 1.5


def build_camera_events(
    cameras: Mapping[str, CameraCalibration],
    *,
    root_path: str = CAMERAS_PATH,
    point_layers: Mapping[str, CameraPointProjection] | None = None,
    box_layers: Mapping[str, CameraBoxProjection] | None = None,
    max_projected_points: int = 10_000,
) -> Sequence[VisualizationEvent]:
    """Build visualization events for all cameras in one sample.

    Each camera gets a ``Transform3D`` (lidar-to-camera extrinsic expressed as
    child-from-parent), a ``Pinhole`` (intrinsic), and an ``Image`` logged
    under ``{root_path}/{cam_name}``.  Rerun uses the transform hierarchy to
    enable automatic point-to-image projection when hovering over 3D points.

    Args:
        cameras: Image file and calibration of every camera, by camera name.
        root_path: Entity path prefix for all camera entities.
        point_layers: Named LiDAR layers to project persistently onto each image.
        box_layers: Named 3D box layers to render as projected wireframes.
        max_projected_points: Maximum points retained per layer and camera.

    Returns:
        Sequence[VisualizationEvent]: A ``Transform3DEvent``, ``PinholeEvent`` and
            ``ImageEvent`` triplet per camera, followed by its projected layers.

    Raises:
        ValueError: If ``max_projected_points`` is not positive.
        FileNotFoundError: If a camera image cannot be read from disk.
    """
    if max_projected_points <= 0:
        raise ValueError("max_projected_points must be greater than zero")

    events: list[VisualizationEvent] = []
    for cam_name, camera in cameras.items():
        cam_path = f"{root_path}/{cam_name}"
        camera_events, intrinsic, extrinsic, resolution = _build_single_camera_events(
            camera, cam_path
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
    camera: CameraCalibration,
    cam_path: str,
) -> tuple[
    Sequence[VisualizationEvent],
    Float32[np.ndarray, "3 3"],
    Float32[np.ndarray, "4 4"],
    tuple[int, int],
]:
    """
    Build the transform, pinhole and image events of one camera.

    Args:
        camera: Image file and calibration of the camera.
        cam_path: Entity path of the camera.

    Returns:
        tuple: The three events, the intrinsics, the LiDAR-to-camera transform and the image
            resolution as width and height.
    """
    image_array = _load_image(camera.image_path)
    height, width = image_array.shape[:2]

    intrinsic = np.asarray(camera.intrinsics, dtype=np.float32).reshape(3, 3)
    extrinsic = np.asarray(camera.lidar_to_camera, dtype=np.float32).reshape(4, 4)

    resolution = (width, height)
    return (
        (
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
        ),
        intrinsic,
        extrinsic,
        resolution,
    )


def _project_points(
    points: Float32[np.ndarray, "num_points 3"],
    intrinsic: Float32[np.ndarray, "3 3"],
    extrinsic: Float32[np.ndarray, "4 4"],
    resolution: tuple[int, int],
    *,
    max_points: int,
) -> tuple[
    Float32[np.ndarray, "num_projected 2"],
    Int64[np.ndarray, " num_projected"],
    Float32[np.ndarray, " num_projected"],
]:
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
    intrinsic: Float32[np.ndarray, "3 3"],
    extrinsic: Float32[np.ndarray, "4 4"],
    resolution: tuple[int, int],
    max_points: int,
) -> Sequence[VisualizationEvent]:
    """Build one persistent point overlay for a camera image."""
    positions = ensure_xyz(layer.points)
    labels = (
        np.asarray(layer.labels, dtype=np.int64).reshape(-1) if layer.labels is not None else None
    )
    if labels is not None and labels.shape[0] != positions.shape[0]:
        raise ValueError(f"Camera point layer {path!r} labels must align with points")
    colors = np.asarray(layer.colors, dtype=np.uint8) if layer.colors is not None else None
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


def _box_corners(box: Float32[np.ndarray, " box_params"]) -> Float32[np.ndarray, "8 3"]:
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
    boxes: Float32[np.ndarray, "num_boxes box_params"],
    intrinsic: Float32[np.ndarray, "3 3"],
    extrinsic: Float32[np.ndarray, "4 4"],
    resolution: tuple[int, int],
) -> tuple[Sequence[Float32[np.ndarray, "2 2"]], Int64[np.ndarray, " num_strips"]]:
    """Project visible edges and return their source box indices."""
    width, height = resolution
    strips: list[Float32[np.ndarray, "2 2"]] = []
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
    intrinsic: Float32[np.ndarray, "3 3"],
    extrinsic: Float32[np.ndarray, "4 4"],
    resolution: tuple[int, int],
) -> Sequence[VisualizationEvent]:
    """Build class-colored projected 3D box wireframes for one camera."""
    boxes = np.asarray(layer.boxes, dtype=np.float32)
    labels = np.asarray(layer.labels, dtype=np.int64).reshape(-1)
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


def _load_image(img_path: str) -> UInt8[np.ndarray, "height width 3"]:
    """Load an RGB image from disk using cv2."""
    # cv2.imread only warns on stderr for a missing file, so check it first.
    image = cv2.imread(img_path, cv2.IMREAD_COLOR) if Path(img_path).is_file() else None
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
