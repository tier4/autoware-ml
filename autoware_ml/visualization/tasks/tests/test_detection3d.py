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

import unittest

import numpy as np

from autoware_ml.visualization.tasks.detection3d import (
    build_detection3d_data_events,
    build_detection3d_events,
    detection_iou_statistics,
    oriented_box_iou_3d,
)
from autoware_ml.visualization.viewers.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    PointCloud3DEvent,
    ScalarEvent,
)

_ONE_BOX = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32)
_ONE_POINT = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
_ONE_SCORE = np.array([0.9], dtype=np.float32)
_ONE_LABEL = np.array([1], dtype=np.int64)


class TestDetection3DEvents(unittest.TestCase):
    """The prediction adapter logs the boxes, the points, the legend and the frame metrics."""

    def test_logs_boxes_and_points(self) -> None:
        """Predicted and ground-truth boxes are logged with the points and scored labels."""
        events = build_detection3d_events(
            pred_boxes=_ONE_BOX,
            pred_scores=_ONE_SCORE,
            pred_labels=_ONE_LABEL,
            points=_ONE_POINT,
            gt_boxes=_ONE_BOX,
            gt_labels=_ONE_LABEL,
            class_names=["pedestrian", "car"],
        )

        self.assertTrue(any(isinstance(event, PointCloud3DEvent) for event in events))
        box_events = [event for event in events if isinstance(event, Boxes3DEvent)]
        self.assertEqual(
            [event.path for event in box_events],
            ["scene/prediction/detections", "scene/ground_truth/detections"],
        )
        self.assertIsNotNone(box_events[0].class_ids)
        self.assertEqual(box_events[0].labels, ["car (0.90)"])

    def test_logs_frame_metrics(self) -> None:
        """A perfectly matching prediction reports one true positive at an IoU of one."""
        events = build_detection3d_events(
            pred_boxes=_ONE_BOX,
            pred_scores=_ONE_SCORE,
            pred_labels=_ONE_LABEL,
            points=None,
            gt_boxes=_ONE_BOX,
            gt_labels=_ONE_LABEL,
        )

        metrics = {event.path: event.value for event in events if isinstance(event, ScalarEvent)}
        self.assertEqual(metrics["scene/metrics/detection/num_predictions"], 1.0)
        self.assertEqual(metrics["scene/metrics/detection/num_ground_truth"], 1.0)
        self.assertAlmostEqual(metrics["scene/metrics/detection/mean_score"], 0.9, places=6)
        self.assertEqual(metrics["scene/metrics/detection/true_positives"], 1.0)
        self.assertAlmostEqual(metrics["scene/metrics/detection/mean_best_iou"], 1.0, places=6)
        self.assertAlmostEqual(metrics["scene/metrics/detection/max_iou"], 1.0, places=6)
        self.assertAlmostEqual(metrics["scene/metrics/detection/mean_matched_iou"], 1.0, places=6)

    def test_legend_covers_every_declared_class(self) -> None:
        """A frame containing one class must still name all classes in the legend."""
        events = build_detection3d_events(
            pred_boxes=_ONE_BOX,
            pred_scores=_ONE_SCORE,
            pred_labels=np.array([0], dtype=np.int64),
            points=None,
            class_names=["pedestrian", "car", "truck"],
        )

        context = next(event for event in events if isinstance(event, AnnotationContextEvent))
        self.assertEqual(
            [annotation.label for annotation in context.annotations],
            ["pedestrian", "car", "truck"],
        )


class TestDetection3DEventsValidation(unittest.TestCase):
    """The prediction adapter rejects inconsistent box sets."""

    def test_rejects_incomplete_ground_truth(self) -> None:
        """Ground-truth boxes and labels must be given together."""
        cases = {"boxes only": (_ONE_BOX, None), "labels only": (None, _ONE_LABEL)}
        for case, (gt_boxes, gt_labels) in cases.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "both be provided or both omitted"):
                    build_detection3d_events(
                        pred_boxes=_ONE_BOX,
                        pred_scores=_ONE_SCORE,
                        pred_labels=_ONE_LABEL,
                        points=None,
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                    )

    def test_rejects_mismatched_lengths(self) -> None:
        """The predicted boxes, scores and labels must hold one entry per box."""
        two_boxes = np.ones((2, 7), dtype=np.float32)
        cases = {
            "labels": (two_boxes, np.array([0.5, 0.4], dtype=np.float32), _ONE_LABEL),
            "scores": (two_boxes, _ONE_SCORE, np.array([0, 1], dtype=np.int64)),
        }
        for case, (pred_boxes, pred_scores, pred_labels) in cases.items():
            with self.subTest(mismatched=case):
                with self.assertRaisesRegex(ValueError, "same length"):
                    build_detection3d_events(
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        pred_labels=pred_labels,
                        points=None,
                    )

    def test_rejects_malformed_boxes(self) -> None:
        """Predicted boxes need the centre, size and yaw columns."""
        with self.assertRaisesRegex(ValueError, r"shape \(N, >=7\)"):
            build_detection3d_events(
                pred_boxes=np.zeros((1, 3), dtype=np.float32),
                pred_scores=_ONE_SCORE,
                pred_labels=_ONE_LABEL,
                points=None,
            )


class TestDetectionIoU(unittest.TestCase):
    """The frame statistics match same-class boxes by their oriented 3D IoU."""

    def test_matches_same_class_boxes_only(self) -> None:
        """Identical boxes of different classes never match."""
        shifted = _ONE_BOX.copy()
        shifted[0, 0] += 20.0
        self.assertAlmostEqual(oriented_box_iou_3d(_ONE_BOX[0], _ONE_BOX[0]), 1.0, places=6)

        stats = detection_iou_statistics(
            np.vstack([_ONE_BOX, shifted]),
            np.array([0, 1], dtype=np.int64),
            _ONE_BOX,
            _ONE_LABEL,
        )

        self.assertEqual(stats["true_positives"], 0.0)
        self.assertEqual(stats["false_positives"], 2.0)
        self.assertEqual(stats["mean_best_iou"], 0.0)
        self.assertEqual(stats["max_iou"], 0.0)

    def test_reports_near_misses_before_thresholding(self) -> None:
        """The best overlap is reported even when no box clears the match threshold."""
        shifted = _ONE_BOX.copy()
        shifted[0, 0] += 2.0

        stats = detection_iou_statistics(shifted, _ONE_LABEL, _ONE_BOX, _ONE_LABEL)

        self.assertEqual(stats["true_positives"], 0.0)
        self.assertEqual(stats["mean_matched_iou"], 0.0)
        self.assertGreater(stats["mean_best_iou"], 0.0)
        self.assertLess(stats["mean_best_iou"], 0.5)
        self.assertAlmostEqual(stats["max_iou"], stats["mean_best_iou"])


class TestDetection3DDataEvents(unittest.TestCase):
    """The data adapter logs the ground truth of a sample previewed without predictions."""

    def test_logs_ground_truth_only(self) -> None:
        """Only the ground-truth boxes are logged, with their class ids."""
        events = build_detection3d_data_events(
            points=_ONE_POINT,
            gt_boxes=_ONE_BOX,
            gt_labels=_ONE_LABEL,
            class_names=["pedestrian", "car"],
        )

        box_events = [event for event in events if isinstance(event, Boxes3DEvent)]
        self.assertEqual([event.path for event in box_events], ["scene/ground_truth/detections"])
        self.assertIsNotNone(box_events[0].class_ids)

    def test_rejects_mismatched_labels(self) -> None:
        """The ground truth must hold one label per box."""
        with self.assertRaisesRegex(ValueError, "same length"):
            build_detection3d_data_events(
                points=None,
                gt_boxes=_ONE_BOX,
                gt_labels=np.array([0, 1], dtype=np.int64),
            )

    def test_rejects_malformed_boxes(self) -> None:
        """Ground-truth boxes need the centre, size and yaw columns."""
        with self.assertRaisesRegex(ValueError, r"shape \(N, >=7\)"):
            build_detection3d_data_events(
                points=None,
                gt_boxes=np.zeros((1, 3), dtype=np.float32),
                gt_labels=np.array([0], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
