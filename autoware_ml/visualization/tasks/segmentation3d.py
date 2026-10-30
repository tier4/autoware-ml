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

"""3D semantic-segmentation visualization adapters."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32, Int64
import numpy as np

from autoware_ml.visualization.colors import (
    build_label_palette,
    labels_to_colors,
    scalar_to_heatmap_colors,
)
from autoware_ml.visualization.common import (
    build_class_annotation_context,
    build_sample_metadata_events,
    ensure_xyz,
    format_class_label,
    resolve_palette_size,
)
from autoware_ml.visualization.entity_paths import SCENE_PATH
from autoware_ml.visualization.sensors.lidar import build_lidar_reference_events
from autoware_ml.visualization.viewers.events import (
    PointCloud3DEvent,
    ScalarEvent,
    VisualizationEvent,
)


def build_segmentation3d_data_events(
    points: Float32[np.ndarray, "num_points num_features"],
    labels: Int64[np.ndarray, " num_points"],
    *,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    root_path: str = SCENE_PATH,
    point_radius: float = 0.04,
    point_labels: bool = False,
    sample_name: str | None = None,
) -> Sequence[VisualizationEvent]:
    """
    Build the events of a segmentation sample previewed without predictions.

    Args:
        points: Points of the sample.
        labels: Ground-truth class of every point.
        class_names: Class names by index, None to show the indices.
        ignore_index: Label drawn as unlabelled.
        root_path: Entity path of the scene.
        point_radius: Radius of every point.
        point_labels: Whether to attach the class name to every point.
        sample_name: Name of the sample, None to log none.

    Returns:
        Sequence[VisualizationEvent]: The legend, the LiDAR layers, the ground-truth points and
            the point count.

    Raises:
        ValueError: If the labels and the points differ in length.
    """
    point_positions = ensure_xyz(points)
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_np.shape[0] != point_positions.shape[0]:
        raise ValueError("labels must have the same length as points")

    palette = build_label_palette(resolve_palette_size([labels_np], class_names))
    radius = float(point_radius)
    events: list[VisualizationEvent] = [*build_sample_metadata_events(root_path, sample_name)]
    semantic_path = f"{root_path}/ground_truth/segmentation"
    annotation_context = build_class_annotation_context(semantic_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)
    events.extend(
        build_lidar_reference_events(points, root_path=root_path, point_radius=point_radius)
    )
    events.append(
        PointCloud3DEvent(
            path=semantic_path,
            positions=point_positions,
            colors=labels_to_colors(labels_np, palette, ignore_index=ignore_index),
            labels=_build_point_labels(labels_np, class_names, point_labels),
            radii=radius,
            class_ids=labels_np,
        )
    )
    events.append(
        ScalarEvent(
            f"{root_path}/metrics/segmentation/num_points",
            float(point_positions.shape[0]),
        )
    )
    return events


def build_segmentation3d_events(
    points: Float32[np.ndarray, "num_points num_features"],
    pred_labels: Int64[np.ndarray, " num_points"],
    *,
    pred_probs: Float32[np.ndarray, "num_points num_classes"] | None = None,
    gt_labels: Int64[np.ndarray, " num_points"] | None = None,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    root_path: str = SCENE_PATH,
    point_radius: float = 0.04,
    point_labels: bool = False,
    sample_name: str | None = None,
) -> Sequence[VisualizationEvent]:
    """
    Build the events of a segmentation sample with predictions.

    With class probabilities, the confidence and the normalized entropy,
    ``-sum(p * log(p)) / log(num_classes)``, are logged per point and averaged per frame.

    Args:
        points: Points the predictions label.
        pred_labels: Predicted class of every point.
        pred_probs: Class probabilities of every point, None to skip the uncertainty layers.
        gt_labels: Ground-truth class of every point, None when unlabelled.
        class_names: Class names by index, None to show the indices.
        ignore_index: Label drawn as unlabelled.
        root_path: Entity path of the scene.
        point_radius: Radius of every point.
        point_labels: Whether to attach the class name to every point.
        sample_name: Name of the sample, None to log none.

    Returns:
        Sequence[VisualizationEvent]: The legends, the LiDAR layers, the predicted and
            ground-truth points, the uncertainty layers and the frame scalars.

    Raises:
        ValueError: If a label or probability array does not match the points.
    """
    point_positions = ensure_xyz(points)
    pred_labels_np = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    if pred_labels_np.shape[0] != point_positions.shape[0]:
        raise ValueError("predicted labels must have the same length as points")

    gt_labels_np = None
    if gt_labels is not None:
        gt_labels_np = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
        if gt_labels_np.shape[0] != point_positions.shape[0]:
            raise ValueError("ground-truth labels must have the same length as points")

    palette = build_label_palette(resolve_palette_size([pred_labels_np, gt_labels_np], class_names))
    radius = float(point_radius)
    events: list[VisualizationEvent] = [*build_sample_metadata_events(root_path, sample_name)]
    prediction_path = f"{root_path}/prediction/segmentation"
    ground_truth_path = f"{root_path}/ground_truth/segmentation"
    for semantic_path in (prediction_path, ground_truth_path):
        annotation_context = build_class_annotation_context(semantic_path, palette, class_names)
        if annotation_context is not None:
            events.append(annotation_context)
    events.extend(
        build_lidar_reference_events(points, root_path=root_path, point_radius=point_radius)
    )
    events.append(
        PointCloud3DEvent(
            path=prediction_path,
            positions=point_positions,
            colors=labels_to_colors(pred_labels_np, palette, ignore_index=ignore_index),
            labels=_build_point_labels(pred_labels_np, class_names, point_labels),
            radii=radius,
            class_ids=pred_labels_np,
        )
    )
    if gt_labels_np is not None:
        events.append(
            PointCloud3DEvent(
                path=ground_truth_path,
                positions=point_positions,
                colors=labels_to_colors(gt_labels_np, palette, ignore_index=ignore_index),
                labels=_build_point_labels(gt_labels_np, class_names, point_labels),
                radii=radius,
                class_ids=gt_labels_np,
            )
        )
    events.append(
        ScalarEvent(
            f"{root_path}/metrics/segmentation/num_points",
            float(pred_labels_np.shape[0]),
        )
    )
    if pred_probs is not None:
        events.extend(
            _build_uncertainty_events(
                pred_probs, point_positions, root_path=root_path, radius=radius
            )
        )
    return events


def _build_uncertainty_events(
    pred_probs: Float32[np.ndarray, "num_points num_classes"],
    point_positions: Float32[np.ndarray, "num_points 3"],
    *,
    root_path: str,
    radius: float,
) -> Sequence[VisualizationEvent]:
    """
    Build the confidence and normalized entropy layers and their frame means.

    Args:
        pred_probs: Class probabilities of every point.
        point_positions: Positions of the points.
        root_path: Entity path of the scene.
        radius: Radius of every point.

    Returns:
        Sequence[VisualizationEvent]: The two frame means and the two per-point layers.

    Raises:
        ValueError: If the probabilities do not hold one row per point and one class at least.
    """
    probabilities = np.asarray(pred_probs, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[0] != point_positions.shape[0]:
        raise ValueError("pred_probs must have shape (N, C) aligned with points")
    if probabilities.shape[1] == 0:
        raise ValueError("pred_probs must contain at least one class")

    confidence = probabilities.max(axis=1).astype(np.float32)
    if probabilities.shape[1] == 1:
        entropy_norm = np.zeros(probabilities.shape[0], dtype=np.float32)
    else:
        entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=1)
        entropy_norm = np.clip(entropy / np.log(probabilities.shape[1]), 0.0, 1.0).astype(
            np.float32
        )
    return (
        ScalarEvent(
            path=f"{root_path}/metrics/segmentation/mean_confidence",
            value=float(confidence.mean()),
        ),
        ScalarEvent(
            path=f"{root_path}/metrics/segmentation/mean_entropy",
            value=float(entropy_norm.mean()),
        ),
        PointCloud3DEvent(
            path=f"{root_path}/prediction/entropy",
            positions=point_positions,
            colors=scalar_to_heatmap_colors(entropy_norm),
            radii=radius,
        ),
        PointCloud3DEvent(
            path=f"{root_path}/prediction/probability",
            positions=point_positions,
            colors=scalar_to_heatmap_colors(confidence),
            radii=radius,
        ),
    )


def _build_point_labels(
    labels: Int64[np.ndarray, " num_points"],
    class_names: Sequence[str] | None,
    enabled: bool,
) -> Sequence[str] | None:
    """
    Build the optional class name text of every point.

    Args:
        labels: Class of every point.
        class_names: Class names by index, None to show the indices.
        enabled: Whether point labels are wanted.

    Returns:
        Sequence[str] | None: One label per point, None when disabled.
    """
    if not enabled:
        return None
    return [format_class_label(int(label), class_names) for label in labels]
