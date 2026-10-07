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

"""Tests for the 3D detection visualization adapter."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.visualization.detection3d import (
    build_detection3d_data_events,
    build_detection3d_events,
    detection_iou_statistics,
    normalize_detection_predictions,
    oriented_box_iou_3d,
)
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    PointCloud3DEvent,
    ScalarEvent,
)

_ONE_BOX = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32)


def test_normalize_detection_predictions_accepts_both_prediction_key_sets() -> None:
    centerpoint_predictions = normalize_detection_predictions(
        {
            "bboxes_3d": np.ones((2, 9), dtype=np.float32),
            "scores_3d": np.array([0.8, 0.7], dtype=np.float32),
            "labels_3d": np.array([0, 1], dtype=np.int64),
        }
    )
    assert centerpoint_predictions["boxes"].shape == (2, 9)

    generic_predictions = normalize_detection_predictions(
        {
            "bboxes": np.ones((1, 7), dtype=np.float32),
            "scores": np.array([0.5], dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
        }
    )
    assert generic_predictions["boxes"].shape == (1, 7)


def test_normalize_detection_predictions_rejects_unknown_key_sets() -> None:
    with pytest.raises(KeyError, match="Expected decoded predictions"):
        normalize_detection_predictions({"pred_labels": np.zeros((1,), dtype=np.int64)})


def test_normalize_detection_predictions_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        normalize_detection_predictions(
            {
                "bboxes": np.ones((2, 7), dtype=np.float32),
                "scores": np.array([0.5], dtype=np.float32),
                "labels": np.array([0], dtype=np.int64),
            }
        )


def test_build_detection3d_events_logs_boxes_and_points() -> None:
    events = build_detection3d_events(
        {
            "bboxes": _ONE_BOX,
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        },
        points=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        gt_boxes=_ONE_BOX,
        gt_labels=np.array([1], dtype=np.int64),
        class_names=["pedestrian", "car"],
    )

    assert any(isinstance(event, PointCloud3DEvent) for event in events)
    box_events = [event for event in events if isinstance(event, Boxes3DEvent)]
    assert [event.path for event in box_events] == [
        "scene/prediction/detections",
        "scene/ground_truth/detections",
    ]
    assert box_events[0].class_ids is not None
    assert box_events[0].labels == ["car (0.90)"]


def test_build_detection3d_events_logs_frame_metrics() -> None:
    events = build_detection3d_events(
        {
            "bboxes": _ONE_BOX,
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        },
        gt_boxes=_ONE_BOX,
        gt_labels=np.array([1], dtype=np.int64),
    )

    metrics = {event.path: event.value for event in events if isinstance(event, ScalarEvent)}
    assert metrics["scene/metrics/detection/num_predictions"] == 1.0
    assert metrics["scene/metrics/detection/num_ground_truth"] == 1.0
    assert metrics["scene/metrics/detection/mean_score"] == pytest.approx(0.9)
    assert metrics["scene/metrics/detection/true_positives"] == 1.0
    assert metrics["scene/metrics/detection/mean_best_iou"] == pytest.approx(1.0)
    assert metrics["scene/metrics/detection/max_iou"] == pytest.approx(1.0)
    assert metrics["scene/metrics/detection/mean_matched_iou"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("gt_boxes", "gt_labels"),
    [(_ONE_BOX, None), (None, np.array([1], dtype=np.int64))],
)
def test_build_detection3d_events_rejects_incomplete_ground_truth(
    gt_boxes: np.ndarray | None,
    gt_labels: np.ndarray | None,
) -> None:
    with pytest.raises(ValueError, match="both be provided or both omitted"):
        build_detection3d_events(
            {
                "bboxes": _ONE_BOX,
                "scores": np.array([0.9], dtype=np.float32),
                "labels": np.array([1], dtype=np.int64),
            },
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )


def test_detection_iou_matches_same_class_boxes_only() -> None:
    shifted = _ONE_BOX.copy()
    shifted[0, 0] += 20.0
    assert oriented_box_iou_3d(_ONE_BOX[0], _ONE_BOX[0]) == pytest.approx(1.0)
    stats = detection_iou_statistics(
        np.vstack([_ONE_BOX, shifted]),
        np.array([0, 1]),
        _ONE_BOX,
        np.array([1]),
    )
    assert stats["true_positives"] == 0.0
    assert stats["false_positives"] == 2.0
    assert stats["mean_best_iou"] == 0.0
    assert stats["max_iou"] == 0.0


def test_detection_iou_reports_near_misses_before_thresholding() -> None:
    shifted = _ONE_BOX.copy()
    shifted[0, 0] += 2.0

    stats = detection_iou_statistics(
        shifted,
        np.array([1]),
        _ONE_BOX,
        np.array([1]),
    )

    assert stats["true_positives"] == 0.0
    assert stats["mean_matched_iou"] == 0.0
    assert 0.0 < stats["mean_best_iou"] < 0.5
    assert stats["max_iou"] == pytest.approx(stats["mean_best_iou"])


def test_build_detection3d_events_legend_covers_every_declared_class() -> None:
    """A frame containing one class must still name all classes in the legend."""
    events = build_detection3d_events(
        {
            "bboxes": _ONE_BOX,
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        },
        class_names=["pedestrian", "car", "truck"],
    )

    context = next(event for event in events if isinstance(event, AnnotationContextEvent))
    assert [annotation.label for annotation in context.annotations] == [
        "pedestrian",
        "car",
        "truck",
    ]


def test_build_detection3d_data_events_logs_ground_truth_only() -> None:
    events = build_detection3d_data_events(
        points=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        gt_boxes=_ONE_BOX,
        gt_labels=np.array([1], dtype=np.int64),
        class_names=["pedestrian", "car"],
    )

    box_events = [event for event in events if isinstance(event, Boxes3DEvent)]
    assert [event.path for event in box_events] == ["scene/ground_truth/detections"]
    assert box_events[0].class_ids is not None


def test_build_detection3d_data_events_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_detection3d_data_events(
            gt_boxes=_ONE_BOX,
            gt_labels=np.array([0, 1], dtype=np.int64),
        )


def test_build_detection3d_data_events_rejects_malformed_boxes() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, >=7\)"):
        build_detection3d_data_events(
            gt_boxes=np.zeros((1, 3), dtype=np.float32),
            gt_labels=np.array([0], dtype=np.int64),
        )
