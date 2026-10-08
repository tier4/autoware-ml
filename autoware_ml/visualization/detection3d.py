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
from typing import Any

import numpy as np

from autoware_ml.visualization.colors import build_label_palette, labels_to_colors
from autoware_ml.visualization.common import (
    as_numpy,
    build_class_annotation_context,
    build_lidar_reference_events,
    build_sample_metadata_events,
    format_class_label,
    resolve_palette_size,
)
from autoware_ml.visualization.events import (
    Boxes3DEvent,
    ScalarEvent,
    VisualizationEvent,
)

#: Key sets that identify decoded 3D detection predictions. Shared with the
#: preview task matcher so both stay in sync.
DETECTION_PREDICTION_KEY_SETS: tuple[frozenset[str], ...] = (
    frozenset({"bboxes_3d", "scores_3d", "labels_3d"}),
    frozenset({"bboxes", "scores", "labels"}),
)


def build_detection3d_data_events(
    *,
    points: Any | None = None,
    gt_boxes: Any,
    gt_labels: Any,
    class_names: Sequence[str] | None = None,
    root_path: str = "scene",
    point_radius: float = 0.04,
    sample_name: str | None = None,
    point_color_mode: str = "semantic",
) -> list[VisualizationEvent]:
    """Build backend-neutral 3D detection events for transformed data only."""
    gt_boxes_np = as_numpy(gt_boxes, np.float32)
    gt_labels_np = as_numpy(gt_labels, np.int64).reshape(-1)
    if gt_boxes_np.ndim != 2 or gt_boxes_np.shape[1] < 7:
        raise ValueError(f"ground-truth boxes must have shape (N, >=7), got {gt_boxes_np.shape}")
    if gt_boxes_np.shape[0] != gt_labels_np.shape[0]:
        raise ValueError("ground-truth boxes and labels must have the same length")

    palette = build_label_palette(resolve_palette_size([gt_labels_np], class_names))
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
    ground_truth_path = f"{root_path}/ground_truth/detections"
    annotation_context = build_class_annotation_context(ground_truth_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)

    if points is not None:
        events.extend(
            build_lidar_reference_events(points, root_path=root_path, point_radius=point_radius)
        )

    gt_label_text = [format_class_label(int(label), class_names) for label in gt_labels_np]
    events.append(
        Boxes3DEvent(
            path=ground_truth_path,
            centers=gt_boxes_np[:, :3],
            sizes=gt_boxes_np[:, 3:6],
            yaws=gt_boxes_np[:, 6],
            colors=labels_to_colors(gt_labels_np, palette),
            labels=gt_label_text,
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


def normalize_detection_predictions(
    predictions: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Normalize decoded 3D predictions into one shared visualization contract."""
    if DETECTION_PREDICTION_KEY_SETS[0] <= predictions.keys():
        boxes = as_numpy(predictions["bboxes_3d"], np.float32)
        scores = as_numpy(predictions["scores_3d"], np.float32).reshape(-1)
        labels = as_numpy(predictions["labels_3d"], np.int64).reshape(-1)
    elif DETECTION_PREDICTION_KEY_SETS[1] <= predictions.keys():
        boxes = as_numpy(predictions["bboxes"], np.float32)
        scores = as_numpy(predictions["scores"], np.float32).reshape(-1)
        labels = as_numpy(predictions["labels"], np.int64).reshape(-1)
    else:
        raise KeyError(
            "Expected decoded predictions with either "
            "('bboxes_3d', 'scores_3d', 'labels_3d') or ('bboxes', 'scores', 'labels')."
        )

    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"decoded boxes must have shape (N, >=7), got {boxes.shape}")
    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
        raise ValueError("decoded boxes, scores, and labels must have the same length")
    return {"boxes": boxes, "scores": scores, "labels": labels}


def build_detection3d_events(
    predictions: Mapping[str, Any],
    *,
    points: Any | None = None,
    gt_boxes: Any | None = None,
    gt_labels: Any | None = None,
    class_names: Sequence[str] | None = None,
    root_path: str = "scene",
    point_radius: float = 0.04,
    sample_name: str | None = None,
    point_color_mode: str = "semantic",
) -> list[VisualizationEvent]:
    """Build backend-neutral 3D detection visualization events for one sample."""
    if (gt_boxes is None) != (gt_labels is None):
        raise ValueError(
            "ground-truth boxes and labels must either both be provided or both omitted"
        )
    normalized_predictions = normalize_detection_predictions(predictions)
    pred_boxes = normalized_predictions["boxes"]
    pred_scores = normalized_predictions["scores"]
    pred_labels = normalized_predictions["labels"]

    gt_labels_np = as_numpy(gt_labels, np.int64).reshape(-1) if gt_labels is not None else None
    palette = build_label_palette(resolve_palette_size([pred_labels, gt_labels_np], class_names))
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
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

    pred_colors = labels_to_colors(pred_labels, palette) if pred_labels.size > 0 else None
    pred_label_text = [
        format_class_label(int(label), class_names, float(score))
        for label, score in zip(pred_labels, pred_scores, strict=False)
    ]
    events.append(
        Boxes3DEvent(
            path=prediction_path,
            centers=pred_boxes[:, :3],
            sizes=pred_boxes[:, 3:6],
            yaws=pred_boxes[:, 6],
            colors=pred_colors,
            labels=pred_label_text,
            class_ids=pred_labels,
        )
    )
    events.append(
        ScalarEvent(
            path=f"{root_path}/metrics/detection/num_predictions",
            value=float(pred_boxes.shape[0]),
        )
    )
    if pred_scores.size > 0:
        events.append(
            ScalarEvent(
                path=f"{root_path}/metrics/detection/mean_score",
                value=float(pred_scores.mean()),
            )
        )

    if gt_boxes is not None and gt_labels_np is not None:
        gt_boxes_np = as_numpy(gt_boxes, np.float32)
        if gt_boxes_np.ndim != 2 or gt_boxes_np.shape[1] < 7:
            raise ValueError(
                f"ground-truth boxes must have shape (N, >=7), got {gt_boxes_np.shape}"
            )
        if gt_boxes_np.shape[0] != gt_labels_np.shape[0]:
            raise ValueError("ground-truth boxes and labels must have the same length")
        gt_label_text = [format_class_label(int(label), class_names) for label in gt_labels_np]
        events.append(
            Boxes3DEvent(
                path=ground_truth_path,
                centers=gt_boxes_np[:, :3],
                sizes=gt_boxes_np[:, 3:6],
                yaws=gt_boxes_np[:, 6],
                colors=labels_to_colors(gt_labels_np, palette),
                labels=gt_label_text,
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
            pred_boxes, pred_labels, gt_boxes_np, gt_labels_np
        ).items():
            events.append(ScalarEvent(f"{root_path}/metrics/detection/{metric_name}", metric_value))

    return events


def _rectangle_corners(box: np.ndarray) -> np.ndarray:
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


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        * 0.5
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Return the scalar cross product for two 2D vectors."""
    return float(first[0] * second[1] - first[1] * second[0])


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    # The corner order is clockwise in the usual x-right/y-up plane; the
    # interior is on the positive side of each directed edge in this layout.
    return _cross_2d(edge_end - edge_start, point - edge_start) >= -1e-6


def _intersection(a: np.ndarray, b: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction, edge = b - a, end - start
    denominator = _cross_2d(direction, edge)
    if abs(denominator) < 1e-8:
        return b.copy()
    return a + (_cross_2d(start - a, edge) / denominator) * direction


def _convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject
    for index, start in enumerate(clip):
        if len(output) == 0:
            break
        end, input_polygon = clip[(index + 1) % len(clip)], output
        points: list[np.ndarray] = []
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


def oriented_box_iou_3d(first: np.ndarray, second: np.ndarray) -> float:
    """Compute IoU for boxes represented as ``x,y,z,w,l,h,yaw``."""
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
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
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
