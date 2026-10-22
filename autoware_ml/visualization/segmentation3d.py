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
from typing import Any

import numpy as np

from autoware_ml.visualization.colors import (
    build_label_palette,
    labels_to_colors,
    scalar_to_heatmap_colors,
)
from autoware_ml.visualization.common import (
    as_numpy,
    build_class_annotation_context,
    build_lidar_reference_events,
    build_sample_metadata_events,
    ensure_xyz,
    format_class_label,
    resolve_palette_size,
)
from autoware_ml.visualization.events import (
    PointCloud3DEvent,
    ScalarEvent,
    VisualizationEvent,
)


def build_segmentation3d_data_events(
    points: Any,
    labels: Any,
    *,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    root_path: str = "scene",
    point_radius: float = 0.04,
    point_labels: bool = False,
    sample_name: str | None = None,
    point_color_mode: str = "semantic",
) -> list[VisualizationEvent]:
    """Build backend-neutral events for transformed segmentation data only."""
    point_positions = ensure_xyz(points)
    labels_np = as_numpy(labels, np.int64).reshape(-1)
    if labels_np.shape[0] != point_positions.shape[0]:
        raise ValueError("labels must have the same length as points")

    palette = build_label_palette(resolve_palette_size([labels_np], class_names))
    radius = float(point_radius)
    label_text = _build_point_labels(labels_np, class_names, point_labels)
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
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
            labels=label_text,
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
    points: Any,
    pred_labels: Any,
    *,
    pred_probs: Any | None = None,
    gt_labels: Any | None = None,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    root_path: str = "scene",
    point_radius: float = 0.04,
    point_labels: bool = False,
    sample_name: str | None = None,
    point_color_mode: str = "semantic",
    pred_logits: Any | None = None,
) -> list[VisualizationEvent]:
    """Build backend-neutral segmentation visualization events for one sample."""
    point_positions = ensure_xyz(points)
    pred_labels_np = as_numpy(pred_labels, np.int64).reshape(-1)
    if pred_labels_np.shape[0] != point_positions.shape[0]:
        raise ValueError("predicted labels must have the same length as points")

    if gt_labels is not None:
        gt_labels_np = as_numpy(gt_labels, np.int64).reshape(-1)
        if gt_labels_np.shape[0] != point_positions.shape[0]:
            raise ValueError("ground-truth labels must have the same length as points")
    else:
        gt_labels_np = None

    palette = build_label_palette(resolve_palette_size([pred_labels_np, gt_labels_np], class_names))
    radius = float(point_radius)
    pred_label_text = _build_point_labels(pred_labels_np, class_names, point_labels)
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
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
            labels=pred_label_text,
            radii=radius,
            class_ids=pred_labels_np,
        )
    )

    if gt_labels_np is not None:
        gt_label_text = _build_point_labels(gt_labels_np, class_names, point_labels)
        events.append(
            PointCloud3DEvent(
                path=ground_truth_path,
                positions=point_positions,
                colors=labels_to_colors(gt_labels_np, palette, ignore_index=ignore_index),
                labels=gt_label_text,
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
    probabilities: np.ndarray | None = None
    if pred_logits is not None:
        logits_np = as_numpy(pred_logits, np.float32)
        if logits_np.ndim != 2 or logits_np.shape[0] != pred_labels_np.shape[0]:
            raise ValueError("pred_logits must have shape (N, C) aligned with points")
        if logits_np.shape[1] == 0:
            raise ValueError("pred_logits must contain at least one class")
        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    elif pred_probs is not None:
        probabilities = as_numpy(pred_probs, np.float32)
        if probabilities.ndim != 2 or probabilities.shape[0] != pred_labels_np.shape[0]:
            raise ValueError("pred_probs must have shape (N, C) aligned with points")
        if probabilities.shape[1] == 0:
            raise ValueError("pred_probs must contain at least one class")

    if probabilities is not None:
        confidence = probabilities.max(axis=1).astype(np.float32)
        if probabilities.shape[1] == 1:
            entropy_norm = np.zeros(probabilities.shape[0], dtype=np.float32)
        else:
            entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=1)
            entropy_norm = np.clip(entropy / np.log(probabilities.shape[1]), 0.0, 1.0).astype(
                np.float32
            )
        events.extend(
            [
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
            ]
        )

    return events


def _build_point_labels(
    labels: np.ndarray,
    class_names: Sequence[str] | None,
    enabled: bool,
) -> list[str] | None:
    """Build optional per-point label text."""
    if not enabled:
        return None
    return [format_class_label(int(label), class_names) for label in labels]
