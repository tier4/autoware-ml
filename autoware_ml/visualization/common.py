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

from jaxtyping import Float32, Int64, Shaped, UInt8
import numpy as np

from autoware_ml.visualization.viewers.events import (
    AnnotationContextEvent,
    AnnotationInfo,
    PointCloud3DEvent,
    TextEvent,
    VisualizationEvent,
)


def ensure_xyz(
    points: Float32[np.ndarray, "num_points num_features"],
) -> Float32[np.ndarray, "num_points 3"]:
    """
    Keep the xyz columns of a point array.

    Args:
        points: Points with at least the x, y and z columns.

    Returns:
        Float32[np.ndarray, "num_points 3"]: The point positions.

    Raises:
        ValueError: If the points have fewer than three columns.
    """
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {array.shape}")
    return array[:, :3]


def build_scene_extent_event(
    points: Float32[np.ndarray, "num_points num_features"] | None,
    boxes: Sequence[Float32[np.ndarray, "num_boxes box_params"]],
    *,
    root_path: str,
) -> PointCloud3DEvent | None:
    """
    Build two invisible corner points spanning everything one frame draws.

    Rerun fits the initial eye of a 3D view to the bounds of the entities that view shows.
    Sibling views of the same frame therefore start from different eyes as soon as one of
    them shows boxes that stick out of the point cloud. Logging the union extent once, and
    listing it in every scene view, gives those views identical bounds and so an identical
    starting eye.

    Args:
        points: Points of the frame, None when it has none.
        boxes: Every box set the frame draws, centre and size in the first six columns.
        root_path: Entity path of the scene.

    Returns:
        PointCloud3DEvent | None: The two corners, None when the frame draws nothing.
    """
    corners: list[Float32[np.ndarray, "num_corners 3"]] = []
    if points is not None:
        xyz = ensure_xyz(points)
        if xyz.shape[0]:
            corners.append(xyz)
    for box_array in boxes:
        array = np.asarray(box_array, dtype=np.float32)
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


def ensure_image_uint8(
    image: Shaped[np.ndarray, "dim_0 dim_1 dim_2"],
) -> UInt8[np.ndarray, "height width 3"]:
    """
    Normalize an HWC or CHW image to an HWC uint8 RGB array.

    Args:
        image: Image with the channels first or last, in [0, 1] when floating point. Channels
            past the third, such as the depth channels of a fused image, are dropped.

    Returns:
        UInt8[np.ndarray, "height width 3"]: The image ready to log.

    Raises:
        ValueError: If the image is not three-dimensional.
    """
    array = np.asarray(image)
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
    """
    Build a readable class label string.

    Args:
        class_id: Class index.
        class_names: Class names by index, None to show the index.
        score: Optional score shown after the name.

    Returns:
        str: The class name, or the index without a name, with the score when given.
    """
    if class_names is not None and 0 <= class_id < len(class_names):
        base = class_names[class_id]
    else:
        base = str(class_id)
    if score is None:
        return base
    return f"{base} ({score:.2f})"


def resolve_palette_size(
    label_arrays: Sequence[Int64[np.ndarray, " num_labels"] | None],
    class_names: Sequence[str] | None = None,
) -> int:
    """
    Return a palette size covering every declared class and observed label.

    Sizing on the declared ``class_names`` as well as the labels present in the current sample
    keeps the viewer legend stable while scrubbing the timeline: without it, a sample containing
    only two of ten classes would publish a two-entry legend and leave the remaining classes
    unnamed.

    Args:
        label_arrays: Label sets of the sample, None for an absent one.
        class_names: Declared class names, None when unknown.

    Returns:
        int: Number of palette entries.
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
    palette: UInt8[np.ndarray, "num_classes 4"],
    class_names: Sequence[str] | None = None,
) -> AnnotationContextEvent | None:
    """
    Build one semantic legend for visualization descendants.

    Args:
        root_path: Entity path the legend applies to, with every descendant.
        palette: RGBA color of every class.
        class_names: Class names by index, None to name the classes by index.

    Returns:
        AnnotationContextEvent | None: The legend, None for an empty palette.
    """
    if palette.shape[0] == 0:
        return None
    return AnnotationContextEvent(
        path=root_path,
        annotations=tuple(
            AnnotationInfo(
                id=index,
                label=format_class_label(index, class_names),
                color=tuple(int(channel) for channel in palette[index]),
            )
            for index in range(palette.shape[0])
        ),
    )


def build_sample_metadata_events(
    root_path: str, sample_name: str | None
) -> Sequence[VisualizationEvent]:
    """
    Build lightweight per-sample metadata events.

    Args:
        root_path: Entity path of the scene.
        sample_name: Name of the sample, None to log nothing.

    Returns:
        Sequence[VisualizationEvent]: The sample name event, if any.
    """
    if sample_name is None:
        return ()
    return (TextEvent(f"{root_path}/meta/sample", sample_name),)
