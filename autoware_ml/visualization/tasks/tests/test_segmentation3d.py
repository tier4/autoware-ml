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

"""Tests for the 3D semantic-segmentation visualization adapter."""

from __future__ import annotations

import unittest

from jaxtyping import Float32
import numpy as np

from autoware_ml.visualization.colors import scalar_to_heatmap_colors
from autoware_ml.visualization.tasks.segmentation3d import (
    build_segmentation3d_data_events,
    build_segmentation3d_events,
)
from autoware_ml.visualization.viewers.events import (
    AnnotationContextEvent,
    PointCloud3DEvent,
    ScalarEvent,
)

_TWO_POINTS = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
_TWO_LABELS = np.array([0, 1], dtype=np.int64)
_CONFIDENT_PROBS = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)


def _softmax(
    logits: Float32[np.ndarray, "num_points num_classes"],
) -> Float32[np.ndarray, "num_points num_classes"]:
    """Return the class probabilities of every row of logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)


class TestSegmentation3DEvents(unittest.TestCase):
    """The prediction adapter logs the predicted and ground-truth points with their legend."""

    def test_logs_prediction_and_ground_truth(self) -> None:
        """Prediction, ground truth and both LiDAR layers are logged with a legend."""
        events = build_segmentation3d_events(
            _TWO_POINTS,
            _TWO_LABELS,
            pred_probs=_CONFIDENT_PROBS,
            gt_labels=np.array([1, 1], dtype=np.int64),
            class_names=["road", "car"],
        )

        point_events = [event for event in events if isinstance(event, PointCloud3DEvent)]
        point_paths = [event.path for event in point_events]
        for path in (
            "scene/prediction/segmentation",
            "scene/ground_truth/segmentation",
            "scene/lidar/intensity",
            "scene/lidar/solid",
        ):
            with self.subTest(path=path):
                self.assertIn(path, point_paths)
        self.assertTrue(any(isinstance(event, AnnotationContextEvent) for event in events))

        prediction = next(e for e in point_events if e.path == "scene/prediction/segmentation")
        self.assertIsNotNone(prediction.class_ids)
        self.assertIsNone(prediction.labels)

    def test_emits_point_labels_on_request(self) -> None:
        """Point labels carry the class name of every point when enabled."""
        events = build_segmentation3d_events(
            _TWO_POINTS,
            _TWO_LABELS,
            class_names=["road", "car"],
            point_labels=True,
        )

        prediction = next(
            event
            for event in events
            if isinstance(event, PointCloud3DEvent)
            and event.path == "scene/prediction/segmentation"
        )
        self.assertEqual(prediction.labels, ["road", "car"])

    def test_legend_covers_every_declared_class(self) -> None:
        """A frame containing two classes must still name all classes in the legend."""
        events = build_segmentation3d_events(
            _TWO_POINTS,
            _TWO_LABELS,
            class_names=["road", "car", "building", "vegetation"],
        )

        context = next(event for event in events if isinstance(event, AnnotationContextEvent))
        self.assertEqual(
            [annotation.label for annotation in context.annotations],
            ["road", "car", "building", "vegetation"],
        )


class TestSegmentation3DUncertainty(unittest.TestCase):
    """Class probabilities add the confidence and entropy layers and their frame means."""

    def test_logs_uncertainty_clouds_and_metrics(self) -> None:
        """Probabilities add the entropy and probability clouds and the frame metrics."""
        events = build_segmentation3d_events(_TWO_POINTS, _TWO_LABELS, pred_probs=_CONFIDENT_PROBS)

        point_paths = [event.path for event in events if isinstance(event, PointCloud3DEvent)]
        self.assertIn("scene/prediction/entropy", point_paths)
        self.assertIn("scene/prediction/probability", point_paths)

        metrics = {event.path: event.value for event in events if isinstance(event, ScalarEvent)}
        self.assertEqual(metrics["scene/metrics/segmentation/num_points"], 2.0)
        self.assertAlmostEqual(metrics["scene/metrics/segmentation/mean_confidence"], 0.9, places=6)
        self.assertGreaterEqual(metrics["scene/metrics/segmentation/mean_entropy"], 0.0)
        self.assertLessEqual(metrics["scene/metrics/segmentation/mean_entropy"], 1.0)

    def test_colors_confidence_and_entropy_from_probabilities(self) -> None:
        """Confident probabilities color the confidence heatmap and give near-zero entropy."""
        probabilities = _softmax(np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32))
        events = build_segmentation3d_events(_TWO_POINTS, _TWO_LABELS, pred_probs=probabilities)

        entropy = next(
            event
            for event in events
            if isinstance(event, PointCloud3DEvent) and event.path.endswith("/entropy")
        )
        self.assertIsNotNone(entropy.colors)
        self.assertEqual(entropy.colors.shape, (2, 4))
        probability = next(
            event
            for event in events
            if isinstance(event, PointCloud3DEvent) and event.path.endswith("/probability")
        )
        np.testing.assert_array_equal(
            probability.colors, scalar_to_heatmap_colors(probabilities.max(axis=1))
        )
        metrics = {event.path: event.value for event in events if isinstance(event, ScalarEvent)}
        self.assertLess(metrics["scene/metrics/segmentation/mean_entropy"], 0.001)
        self.assertGreater(metrics["scene/metrics/segmentation/mean_confidence"], 0.999)

    def test_normalizes_maximum_entropy_to_one(self) -> None:
        """Uniform probabilities give a normalized entropy of one."""
        events = build_segmentation3d_events(
            _TWO_POINTS,
            np.array([0, 0], dtype=np.int64),
            pred_probs=np.full((2, 4), 0.25, dtype=np.float32),
        )

        metrics = {event.path: event.value for event in events if isinstance(event, ScalarEvent)}
        self.assertAlmostEqual(metrics["scene/metrics/segmentation/mean_entropy"], 1.0, places=6)
        self.assertAlmostEqual(
            metrics["scene/metrics/segmentation/mean_confidence"], 0.25, places=6
        )


class TestSegmentation3DEventsValidation(unittest.TestCase):
    """The prediction adapter rejects arrays that do not match the points."""

    def test_rejects_misaligned_predictions(self) -> None:
        """The predicted labels must hold one label per point."""
        with self.assertRaisesRegex(ValueError, "predicted labels must have the same length"):
            build_segmentation3d_events(_TWO_POINTS, np.array([0], dtype=np.int64))

    def test_rejects_misaligned_ground_truth(self) -> None:
        """The ground-truth labels must hold one label per point."""
        with self.assertRaisesRegex(ValueError, "ground-truth labels must have the same length"):
            build_segmentation3d_events(
                _TWO_POINTS,
                _TWO_LABELS,
                gt_labels=np.array([0], dtype=np.int64),
            )

    def test_rejects_misaligned_probabilities(self) -> None:
        """The probabilities must hold one row per point."""
        with self.assertRaisesRegex(ValueError, r"pred_probs must have shape \(N, C\)"):
            build_segmentation3d_events(
                _TWO_POINTS,
                _TWO_LABELS,
                pred_probs=np.zeros((1, 2), dtype=np.float32),
            )


class TestSegmentation3DDataEvents(unittest.TestCase):
    """The data adapter logs the ground truth of a sample previewed without predictions."""

    def test_logs_single_data_cloud(self) -> None:
        """The LiDAR layers and one ground-truth cloud are logged with a legend."""
        events = build_segmentation3d_data_events(
            _TWO_POINTS,
            _TWO_LABELS,
            class_names=["road", "car"],
        )

        point_events = [event for event in events if isinstance(event, PointCloud3DEvent)]
        self.assertEqual(
            [event.path for event in point_events],
            ["scene/lidar/solid", "scene/lidar/intensity", "scene/ground_truth/segmentation"],
        )
        self.assertTrue(any(isinstance(event, AnnotationContextEvent) for event in events))

    def test_rejects_misaligned_labels(self) -> None:
        """The labels must hold one label per point."""
        with self.assertRaisesRegex(ValueError, "labels must have the same length as points"):
            build_segmentation3d_data_events(_TWO_POINTS, np.array([0], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
