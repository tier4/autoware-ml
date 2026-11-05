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

"""Tests for the calibration-status visualization adapter."""

from __future__ import annotations

import unittest

import numpy as np

from autoware_ml.utils.calibration import CalibrationStatus
from autoware_ml.visualization.tasks.calibration_status import build_calibration_status_events
from autoware_ml.visualization.tests.doubles import calibration_matrices
from autoware_ml.visualization.viewers.events import (
    ImageEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextDocumentEvent,
    TextEvent,
    Transform3DEvent,
)

_IN_FRONT_OF_CAMERA = np.array([[0.0, 0.0, 10.0, 0.4]], dtype=np.float32)
_FUSED_IMAGE = np.zeros((720, 1280, 5), dtype=np.float32)
_FUSED_IMAGE[100, 100, 3] = 0.5
_FUSED_IMAGE[200, 200, 4] = 0.8


class TestCalibrationStatusLayers(unittest.TestCase):
    """The adapter logs the camera frame, the images, the overlays and the lidar points."""

    def test_includes_frames_and_status(self) -> None:
        """A full sample logs the camera, the lidar, both statuses and the fused overlays."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            points=_IN_FRONT_OF_CAMERA,
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            fused_image=_FUSED_IMAGE,
            gt_status=CalibrationStatus.CALIBRATED.value,
            pred_status=CalibrationStatus.MISCALIBRATED.value,
            pred_score=0.9,
            sample_name="sample-1",
        )
        paths = {event.path for event in events}

        for path in (
            "calibration_status/camera",
            "calibration_status/lidar/points",
            "calibration_status/status/gt_label",
            "calibration_status/status/pred_label",
            "calibration_status/camera/fused",
            "calibration_status/camera/fused/depth",
            "calibration_status/camera/fused/intensity",
        ):
            with self.subTest(path=path):
                self.assertIn(path, paths)
        fused_overlays = {
            event.path: event
            for event in events
            if isinstance(event, Points2DEvent)
            and event.path.startswith("calibration_status/camera/fused/")
        }
        self.assertEqual(
            fused_overlays["calibration_status/camera/fused/depth"].positions.shape, (1, 2)
        )
        self.assertEqual(
            fused_overlays["calibration_status/camera/fused/intensity"].positions.shape, (1, 2)
        )
        self.assertTrue(
            any(
                isinstance(event, Points2DEvent)
                and event.path == "calibration_status/camera/image/projected_points"
                for event in events
            )
        )
        self.assertTrue(
            any(
                isinstance(event, ScalarEvent)
                and event.path == "calibration_status/status/pred_score"
                for event in events
            )
        )

    def test_logs_camera_geometry_with_an_image(self) -> None:
        """An image adds the pinhole of its resolution, the image and the camera transform."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            image=np.zeros((36, 64, 3), dtype=np.uint8),
        )

        pinhole = next(event for event in events if isinstance(event, PinholeEvent))
        self.assertEqual(pinhole.resolution, (64, 36))
        self.assertTrue(
            any(
                isinstance(event, ImageEvent) and event.path == "calibration_status/camera/image"
                for event in events
            )
        )
        self.assertTrue(any(isinstance(event, Transform3DEvent) for event in events))

    def test_needs_only_the_camera_matrices(self) -> None:
        """Without points or images the adapter still logs the camera frame."""
        events = build_calibration_status_events(**calibration_matrices())

        self.assertEqual([event.path for event in events], ["calibration_status/camera"])
        self.assertIsInstance(events[0], Transform3DEvent)

    def test_skips_projection_behind_the_camera(self) -> None:
        """Points with non-positive depth produce no image overlay."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            points=np.array([[0.0, 0.0, -50.0, 0.4]], dtype=np.float32),
            image=np.zeros((36, 64, 3), dtype=np.uint8),
        )

        self.assertFalse(any(isinstance(event, Points2DEvent) for event in events))
        self.assertTrue(
            any(
                isinstance(event, PointCloud3DEvent)
                and event.path == "calibration_status/lidar/points"
                for event in events
            )
        )

    def test_honors_the_root_path(self) -> None:
        """Every event is logged under the requested root path."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            sample_name="sample-1",
            root_path="dataset/calibration_status",
        )

        for event in events:
            with self.subTest(path=event.path):
                self.assertTrue(event.path.startswith("dataset/calibration_status"))


class TestCalibrationStatusVerdict(unittest.TestCase):
    """The adapter summarizes the predicted and ground-truth statuses as text."""

    def test_summarizes_prediction_and_ground_truth(self) -> None:
        """The one-line summary names the prediction with its score and the ground truth."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            gt_status=CalibrationStatus.CALIBRATED.value,
            pred_status=CalibrationStatus.MISCALIBRATED.value,
            pred_score=0.9,
        )

        summary = next(
            event
            for event in events
            if isinstance(event, TextEvent) and event.path == "calibration_status/status/summary"
        )
        self.assertEqual(summary.text, "pred: miscalibrated (0.90) | gt: calibrated")

    def test_states_the_verdict_as_a_document(self) -> None:
        """The verdict repeats both statuses as headings and says whether they agree."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            fused_image=_FUSED_IMAGE,
            gt_status=CalibrationStatus.CALIBRATED.value,
            pred_status=CalibrationStatus.MISCALIBRATED.value,
            pred_score=0.9,
        )

        verdict = next(
            event
            for event in events
            if isinstance(event, TextDocumentEvent)
            and event.path == "calibration_status/status/verdict"
        )
        self.assertTrue(verdict.markdown)
        self.assertEqual(verdict.text.splitlines()[0], "# MISMATCH")
        self.assertTrue(
            verdict.text.rstrip().endswith(
                "## Prediction: MISCALIBRATED (0.90) · Ground truth: CALIBRATED"
            )
        )

    def test_reports_a_match(self) -> None:
        """Agreeing statuses make the verdict headline a match."""
        events = build_calibration_status_events(
            **calibration_matrices(),
            fused_image=_FUSED_IMAGE,
            gt_status=CalibrationStatus.CALIBRATED.value,
            pred_status=CalibrationStatus.CALIBRATED.value,
            pred_score=0.75,
        )

        verdict = next(event for event in events if isinstance(event, TextDocumentEvent))
        self.assertEqual(verdict.text.splitlines()[0], "# MATCH")


if __name__ == "__main__":
    unittest.main()
