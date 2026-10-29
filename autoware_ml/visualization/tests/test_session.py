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

"""Tests for the public visualization session facade."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
from jaxtyping import Float32
import numpy as np

from autoware_ml.visualization.sensors.camera import CameraCalibration, CameraPointProjection
from autoware_ml.visualization.session import VisualizationSession
from autoware_ml.visualization.tests.doubles import RecordingBackend, calibration_matrices
from autoware_ml.visualization.viewers.contracts import VisualizationSessionConfig
from autoware_ml.visualization.viewers.events import (
    BlueprintEvent,
    Boxes3DEvent,
    ClearEvent,
    ImageEvent,
    PointCloud3DEvent,
    Points2DEvent,
    Transform3DEvent,
    ViewSpec,
)


def _two_points() -> Float32[np.ndarray, "2 4"]:
    """Return two LiDAR points with an intensity column."""
    return np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def _log_empty_detection(session: VisualizationSession) -> None:
    """Log a detection sample without points, predictions or ground truth."""
    session.log_detection3d(
        pred_boxes=np.zeros((0, 7), dtype=np.float32),
        pred_scores=np.zeros((0,), dtype=np.float32),
        pred_labels=np.zeros((0,), dtype=np.int64),
        points=None,
    )


def _write_camera(image_path: Path, focal: float) -> CameraCalibration:
    """Write a 64x36 image and return a camera at the LiDAR origin centred on it."""
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    return CameraCalibration(
        image_path=str(image_path),
        intrinsics=np.array(
            [[focal, 0.0, 32.0], [0.0, focal, 18.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        lidar_to_camera=np.eye(4, dtype=np.float32),
    )


class TestSessionTimeline(unittest.TestCase):
    """The session forwards timeline steps, frames and raw events to its backend."""

    def setUp(self) -> None:
        """Create a session over a recording backend."""
        self.backend = RecordingBackend()
        self.session = VisualizationSession(self.backend)

    def test_forwards_steps_and_events(self) -> None:
        """Steps and task events reach the backend."""
        self.session.set_step(7)
        _log_empty_detection(self.session)

        self.assertEqual(self.backend.steps, [7])
        self.assertTrue(self.backend.events)

    def test_accepts_events_from_an_external_task_adapter(self) -> None:
        """Events built outside the session are forwarded unchanged."""
        blueprint = BlueprintEvent(
            layout=ViewSpec(
                kind="spatial3d",
                name="New task",
                origin="new_task",
                contents=("new_task/output",),
            )
        )

        self.session.log_events([blueprint])

        self.assertEqual(self.backend.events, [blueprint])

    def test_begins_replacement_frame_and_forwards_timestamp(self) -> None:
        """A new frame sets the step and timestamp and recursively clears the scene layers."""
        self.session.begin_frame(70, timestamp=12.5)

        self.assertEqual(self.backend.steps, [70])
        self.assertEqual(self.backend.timestamps, [12.5])
        self.assertEqual(
            self.backend.paths_of(ClearEvent),
            [
                "scene/lidar",
                "scene/prediction",
                "scene/ground_truth",
                "scene/cameras",
                "scene/meta",
            ],
        )
        self.assertTrue(all(event.recursive for event in self.backend.events))

    def test_logs_the_scene_extent_only_when_there_is_geometry(self) -> None:
        """A frame with geometry logs its extent and an empty frame logs nothing."""
        self.session.log_scene_extent(
            np.zeros((2, 3), dtype=np.float32),
            boxes=(np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32),),
        )
        self.session.log_scene_extent(None, boxes=())

        self.assertEqual(self.backend.paths_of(PointCloud3DEvent), ["scene/meta/extent"])
        self.assertEqual(len(self.backend.events), 1)


class TestSessionFromConfig(unittest.TestCase):
    """The session builds the backend its configuration names."""

    def test_builds_the_configured_backend(self) -> None:
        """A no-op backend session accepts steps and task samples."""
        session = VisualizationSession.from_config(VisualizationSessionConfig(backend="noop"))

        session.set_step(0)
        _log_empty_detection(session)


class TestSessionTaskLogging(unittest.TestCase):
    """The task helpers log their samples under the scene entity paths."""

    def setUp(self) -> None:
        """Create a session over a recording backend."""
        self.backend = RecordingBackend()
        self.session = VisualizationSession(self.backend)

    def test_logs_detection_ground_truth(self) -> None:
        """Ground-truth boxes are logged under the ground-truth detections path."""
        self.session.log_detection3d_data(
            points=_two_points(),
            gt_boxes=np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
            gt_labels=np.array([1], dtype=np.int64),
            class_names=["pedestrian", "car"],
        )

        self.assertEqual(self.backend.paths_of(Boxes3DEvent), ["scene/ground_truth/detections"])

    def test_logs_segmentation_predictions(self) -> None:
        """Predictions, ground truth and the LiDAR intensity are logged as point clouds."""
        self.session.log_segmentation3d(
            _two_points(),
            np.array([0, 1], dtype=np.int64),
            gt_labels=np.array([1, 1], dtype=np.int64),
            class_names=["road", "car"],
        )

        point_paths = self.backend.paths_of(PointCloud3DEvent)
        for path in (
            "scene/prediction/segmentation",
            "scene/ground_truth/segmentation",
            "scene/lidar/intensity",
        ):
            with self.subTest(path=path):
                self.assertIn(path, point_paths)

    def test_logs_segmentation_data(self) -> None:
        """Dataset labels are logged under the ground-truth segmentation path."""
        self.session.log_segmentation3d_data(
            _two_points(),
            np.array([0, 1], dtype=np.int64),
            class_names=["road", "car"],
        )

        self.assertIn("scene/ground_truth/segmentation", self.backend.paths_of(PointCloud3DEvent))

    def test_logs_calibration_status(self) -> None:
        """A calibration-status sample logs the camera transform under its own root."""
        self.session.log_calibration_status(**calibration_matrices(), sample_name="sample-1")

        self.assertEqual(self.backend.paths_of(Transform3DEvent), ["calibration_status/camera"])


class TestSessionCameras(unittest.TestCase):
    """The camera helper logs every image and its projected layers."""

    def setUp(self) -> None:
        """Create a session over a recording backend and a temporary image directory."""
        self.backend = RecordingBackend()
        self.session = VisualizationSession(self.backend)
        self.tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_logs_multiview_cameras(self) -> None:
        """Every camera image is logged under its camera path."""
        camera = _write_camera(self.tmp_path / "cam.png", focal=1000.0)

        self.session.log_cameras({"CAM_FRONT": camera, "CAM_BACK": camera})

        self.assertEqual(
            self.backend.paths_of(ImageEvent),
            ["scene/cameras/CAM_FRONT", "scene/cameras/CAM_BACK"],
        )

    def test_logs_persistent_camera_projection_layers(self) -> None:
        """A point layer is projected under the camera's projected path."""
        camera = _write_camera(self.tmp_path / "cam.png", focal=10.0)

        self.session.log_cameras(
            {"CAM_FRONT": camera},
            point_layers={
                "prediction/segmentation": CameraPointProjection(
                    points=np.array([[0.0, 0.0, 10.0]], dtype=np.float32),
                    labels=np.array([0], dtype=np.int64),
                )
            },
        )

        self.assertEqual(
            self.backend.paths_of(Points2DEvent),
            ["scene/cameras/CAM_FRONT/projected/prediction/segmentation"],
        )


if __name__ == "__main__":
    unittest.main()
