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

"""3D detection visualization adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jaxtyping import Float32, Int64
import numpy as np

from autoware_ml.visualization.colors import build_label_palette, labels_to_colors
from autoware_ml.visualization.common import (
    build_class_annotation_context,
    build_sample_metadata_events,
    format_class_label,
    resolve_palette_size,
)
from autoware_ml.visualization.entity_paths import SCENE_PATH
from autoware_ml.visualization.sensors.lidar import build_lidar_reference_events
from autoware_ml.visualization.viewers.events import (
    Boxes3DEvent,
    ScalarEvent,
    VisualizationEvent,
)


def build_detection3d_data_events(
    *,
    points: Float32[np.ndarray, "num_points num_features"] | None,
    gt_boxes: Float32[np.ndarray, "num_boxes box_params"],
    gt_labels: Int64[np.ndarray, " num_boxes"],
    class_names: Sequence[str] | None = None,
    root_path: str = SCENE_PATH,
    point_radius: float = 0.04,
    sample_name: str | None = None,
) -> Sequence[VisualizationEvent]:
    """
    Build the events of a detection sample previewed without predictions.

    Args:
        points: Points of the sample, None to draw the boxes alone.
        gt_boxes: Ground-truth boxes, centre, size and yaw in the first seven columns.
        gt_labels: Class of every ground-truth box.
        class_names: Class names by index, None to show the indices.
        root_path: Entity path of the scene.
        point_radius: Radius of every point.
        sample_name: Name of the sample, None to log none.

    Returns:
        Sequence[VisualizationEvent]: The legend, the LiDAR layers, the ground-truth boxes and
            their count.
    """
    gt_boxes_np, gt_labels_np = _validated_boxes(gt_boxes, gt_labels, "ground-truth")
    palette = build_label_palette(resolve_palette_size([gt_labels_np], class_names))
    events: list[VisualizationEvent] = [*build_sample_metadata_events(root_path, sample_name)]
    ground_truth_path = f"{root_path}/ground_truth/detections"
    annotation_context = build_class_annotation_context(ground_truth_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)
    if points is not None:
        events.extend(
            build_lidar_reference_events(points, root_path=root_path, point_radius=point_radius)
        )
    events.append(
        Boxes3DEvent(
            path=ground_truth_path,
            centers=gt_boxes_np[:, :3],
            sizes=gt_boxes_np[:, 3:6],
            yaws=gt_boxes_np[:, 6],
            colors=labels_to_colors(gt_labels_np, palette),
            labels=[format_class_label(int(label), class_names) for label in gt_labels_np],
            class_ids=gt_labels_np,
        )
    )
    events.append(
        ScalarEvent(
            path=f"{root_path}/metrics/detection/num_ground_truth",
            value=float(gt_boxes_np.shape[0]),
        )
    )
    return events


def build_detection3d_events(
    *,
    pred_boxes: Float32[np.ndarray, "num_predictions box_params"],
    pred_scores: Float32[np.ndarray, " num_predictions"],
    pred_labels: Int64[np.ndarray, " num_predictions"],
    points: Float32[np.ndarray, "num_points num_features"] | None,
    gt_boxes: Float32[np.ndarray, "num_boxes box_params"] | None = None,
    gt_labels: Int64[np.ndarray, " num_boxes"] | None = None,
    class_names: Sequence[str] | None = None,
    root_path: str = SCENE_PATH,
    point_radius: float = 0.04,
    sample_name: str | None = None,
) -> Sequence[VisualizationEvent]:
    """
    Build the events of a detection sample with predictions.

    With ground truth, the boxes are matched by class at a 3D IoU of 0.5 and the frame
    statistics of :func:`detection_iou_statistics` are logged.

    Args:
        pred_boxes: Predicted boxes, centre, size and yaw in the first seven columns.
        pred_scores: Score of every predicted box.
        pred_labels: Class of every predicted box.
        points: Points of the sample, None to draw the boxes alone.
        gt_boxes: Ground-truth boxes, None when the sample is unlabelled.
        gt_labels: Class of every ground-truth box, None when the sample is unlabelled.
        class_names: Class names by index, None to show the indices.
        root_path: Entity path of the scene.
        point_radius: Radius of every point.
        sample_name: Name of the sample, None to log none.

    Returns:
        Sequence[VisualizationEvent]: The legends, the LiDAR layers, the predicted and
            ground-truth boxes, and the frame statistics.

    Raises:
        ValueError: If only one of the ground-truth boxes and labels is given, or the boxes,
            scores and labels of a set differ in length.
    """
    if (gt_boxes is None) != (gt_labels is None):
        raise ValueError(
            "ground-truth boxes and labels must either both be provided or both omitted"
        )
    boxes, labels = _validated_boxes(pred_boxes, pred_labels, "predicted")
    scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError("decoded boxes, scores, and labels must have the same length")
    gt_boxes_np = gt_labels_np = None
    if gt_boxes is not None and gt_labels is not None:
        gt_boxes_np, gt_labels_np = _validated_boxes(gt_boxes, gt_labels, "ground-truth")

    palette = build_label_palette(resolve_palette_size([labels, gt_labels_np], class_names))
    events: list[VisualizationEvent] = [*build_sample_metadata_events(root_path, sample_name)]
    prediction_path = f"{root_path}/prediction/detections"
    ground_truth_path = f"{root_path}/ground_truth/detections"
    for detections_path in (prediction_path, ground_truth_path):
        annotation_context = build_class_annotation_context(detections_path, palette, class_names)
        if annotation_context is not None:
            events.append(annotation_context)
    if points is not None:
        events.extend(
            build_lidar_reference_events(points, root_path=root_path, point_radius=point_radius)
        )

    events.append(
        Boxes3DEvent(
            path=prediction_path,
            centers=boxes[:, :3],
            sizes=boxes[:, 3:6],
            yaws=boxes[:, 6],
            colors=labels_to_colors(labels, palette) if labels.size > 0 else None,
            labels=[
                format_class_label(int(label), class_names, float(score))
                for label, score in zip(labels, scores, strict=True)
            ],
            class_ids=labels,
        )
    )
    events.append(
        ScalarEvent(
            path=f"{root_path}/metrics/detection/num_predictions",
            value=float(boxes.shape[0]),
        )
    )
    if scores.size > 0:
        events.append(
            ScalarEvent(
                path=f"{root_path}/metrics/detection/mean_score",
                value=float(scores.mean()),
            )
        )

    if gt_boxes_np is not None and gt_labels_np is not None:
        events.append(
            Boxes3DEvent(
                path=ground_truth_path,
                centers=gt_boxes_np[:, :3],
                sizes=gt_boxes_np[:, 3:6],
                yaws=gt_boxes_np[:, 6],
                colors=labels_to_colors(gt_labels_np, palette),
                labels=[format_class_label(int(label), class_names) for label in gt_labels_np],
                class_ids=gt_labels_np,
            )
        )
        events.append(
            ScalarEvent(
                path=f"{root_path}/metrics/detection/num_ground_truth",
                value=float(gt_boxes_np.shape[0]),
            )
        )
        for metric_name, metric_value in detection_iou_statistics(
            boxes, labels, gt_boxes_np, gt_labels_np
        ).items():
            events.append(ScalarEvent(f"{root_path}/metrics/detection/{metric_name}", metric_value))
    return events


def _validated_boxes(
    boxes: Float32[np.ndarray, "num_boxes box_params"],
    labels: Int64[np.ndarray, " num_boxes"],
    kind: str,
) -> tuple[Float32[np.ndarray, "num_boxes box_params"], Int64[np.ndarray, " num_boxes"]]:
    """
    Check that a box set holds centre, size and yaw and one label per box.

    Args:
        boxes: Boxes, centre, size and yaw in the first seven columns.
        labels: Class of every box.
        kind: Name of the box set for the error message.

    Returns:
        tuple: The boxes as float32 and the labels as a flat int64 array.

    Raises:
        ValueError: If the boxes have fewer than seven columns or the counts differ.
    """
    boxes_np = np.asarray(boxes, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    if boxes_np.ndim != 2 or boxes_np.shape[1] < 7:
        raise ValueError(f"{kind} boxes must have shape (N, >=7), got {boxes_np.shape}")
    if boxes_np.shape[0] != labels_np.shape[0]:
        raise ValueError(f"{kind} boxes and labels must have the same length")
    return boxes_np, labels_np


def _rectangle_corners(box: Float32[np.ndarray, " box_params"]) -> Float32[np.ndarray, "4 2"]:
    """Return the four bird's-eye-view corners of a yaw-only box."""
    half_width, half_length = box[3] * 0.5, box[4] * 0.5
    corners = np.array(
        [
            [half_width, half_length],
            [-half_width, half_length],
            [-half_width, -half_length],
            [half_width, -half_length],
        ],
        dtype=np.float32,
    )
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    return corners @ np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32) + box[:2]


def _polygon_area(polygon: Float32[np.ndarray, "num_vertices 2"]) -> float:
    """Return the area of a simple polygon by the shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    return float(
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        * 0.5
    )


def _cross_2d(first: Float32[np.ndarray, " 2"], second: Float32[np.ndarray, " 2"]) -> float:
    """Return the scalar cross product for two 2D vectors."""
    return float(first[0] * second[1] - first[1] * second[0])


def _inside(
    point: Float32[np.ndarray, " 2"],
    edge_start: Float32[np.ndarray, " 2"],
    edge_end: Float32[np.ndarray, " 2"],
) -> bool:
    """Return whether a point lies on the inner side of a directed clip edge."""
    # The corner order is clockwise in the usual x-right/y-up plane; the
    # interior is on the positive side of each directed edge in this layout.
    return _cross_2d(edge_end - edge_start, point - edge_start) >= -1e-6


def _intersection(
    a: Float32[np.ndarray, " 2"],
    b: Float32[np.ndarray, " 2"],
    start: Float32[np.ndarray, " 2"],
    end: Float32[np.ndarray, " 2"],
) -> Float32[np.ndarray, " 2"]:
    """Return where segment ``a``-``b`` crosses the line through ``start`` and ``end``."""
    direction, edge = b - a, end - start
    denominator = _cross_2d(direction, edge)
    if abs(denominator) < 1e-8:
        return b.copy()
    return a + (_cross_2d(start - a, edge) / denominator) * direction


def _convex_intersection(
    subject: Float32[np.ndarray, "num_subject_vertices 2"],
    clip: Float32[np.ndarray, "num_clip_vertices 2"],
) -> Float32[np.ndarray, "num_vertices 2"]:
    """Clip one convex polygon by another (Sutherland-Hodgman)."""
    output = subject
    for index, start in enumerate(clip):
        if len(output) == 0:
            break
        end, input_polygon = clip[(index + 1) % len(clip)], output
        points: list[Float32[np.ndarray, " 2"]] = []
        previous = input_polygon[-1]
        for current in input_polygon:
            current_inside = _inside(current, start, end)
            previous_inside = _inside(previous, start, end)
            if current_inside != previous_inside:
                points.append(_intersection(previous, current, start, end))
            if current_inside:
                points.append(current)
            previous = current
        output = np.asarray(points, dtype=np.float32)
    return output


def oriented_box_iou_3d(
    first: Float32[np.ndarray, " box_params"], second: Float32[np.ndarray, " box_params"]
) -> float:
    """
    Compute the 3D IoU of two yaw-only boxes.

    Args:
        first: Box as ``x, y, z, length, width, height, yaw``.
        second: Box as ``x, y, z, length, width, height, yaw``.

    Returns:
        float: Intersection over union of the two boxes.
    """
    bev = _polygon_area(_convex_intersection(_rectangle_corners(first), _rectangle_corners(second)))
    z_overlap = max(
        0.0,
        min(first[2] + first[5] * 0.5, second[2] + second[5] * 0.5)
        - max(first[2] - first[5] * 0.5, second[2] - second[5] * 0.5),
    )
    intersection = bev * z_overlap
    first_volume, second_volume = (
        max(0.0, float(np.prod(first[3:6]))),
        max(0.0, float(np.prod(second[3:6]))),
    )
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0.0 else 0.0


def detection_iou_statistics(
    pred_boxes: Float32[np.ndarray, "num_predictions box_params"],
    pred_labels: Int64[np.ndarray, " num_predictions"],
    gt_boxes: Float32[np.ndarray, "num_boxes box_params"],
    gt_labels: Int64[np.ndarray, " num_boxes"],
    *,
    iou_threshold: float = 0.5,
) -> Mapping[str, float]:
    """Greedily match same-class boxes and return frame-level statistics.

    Besides threshold-qualified detection metrics, report the best overlap for
    every GT box before applying the threshold.  This keeps the quality signal
    informative when a model has near misses but no true positives yet.
    """
    candidates = [
        (oriented_box_iou_3d(pred_boxes[p], gt_boxes[g]), p, g)
        for p in range(len(pred_boxes))
        for g in range(len(gt_boxes))
        if pred_labels[p] == gt_labels[g]
    ]
    matched: list[tuple[float, int, int]] = []
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    for iou, prediction_index, ground_truth_index in sorted(candidates, reverse=True):
        if (
            iou >= iou_threshold
            and prediction_index not in used_predictions
            and ground_truth_index not in used_ground_truth
        ):
            matched.append((iou, prediction_index, ground_truth_index))
            used_predictions.add(prediction_index)
            used_ground_truth.add(ground_truth_index)
    true_positives = len(matched)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives
    best_iou_per_ground_truth = [
        max(
            (iou for iou, _, candidate_ground_truth in candidates if candidate_ground_truth == g),
            default=0.0,
        )
        for g in range(len(gt_boxes))
    ]
    return {
        "iou_threshold": float(iou_threshold),
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "precision": true_positives / len(pred_boxes) if len(pred_boxes) else 0.0,
        "recall": true_positives / len(gt_boxes) if len(gt_boxes) else 0.0,
        "mean_best_iou": (
            float(np.mean(best_iou_per_ground_truth)) if best_iou_per_ground_truth else 0.0
        ),
        "max_iou": max((item[0] for item in candidates), default=0.0),
        "mean_matched_iou": float(np.mean([item[0] for item in matched])) if matched else 0.0,
    }
