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

import cv2
import numpy as np

from autoware_ml.tests.visualization.conftest import RecordingBackend
from autoware_ml.utils.calibration import CalibrationData
from autoware_ml.visualization.cameras import CameraPointProjection
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
    Boxes3DEvent,
    BlueprintEvent,
    ClearEvent,
    ImageEvent,
    PointCloud3DEvent,
    Points2DEvent,
    Transform3DEvent,
    ViewSpec,
)
from autoware_ml.visualization.session import VisualizationSession

_EMPTY_DETECTION = {
    "bboxes": np.zeros((0, 7), dtype=np.float32),
    "scores": np.zeros((0,), dtype=np.float32),
    "labels": np.zeros((0,), dtype=np.int64),
}
_TWO_POINTS = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def test_session_forwards_steps_and_events(recording_backend: RecordingBackend) -> None:
    session = VisualizationSession(recording_backend)

    session.set_step(7)
    session.log_detection3d(_EMPTY_DETECTION)

    assert recording_backend.steps == [7]
    assert recording_backend.events


def test_session_accepts_events_from_an_external_task_adapter(
    recording_backend: RecordingBackend,
) -> None:
    session = VisualizationSession(recording_backend)
    blueprint = BlueprintEvent(
        layout=ViewSpec(
            kind="spatial3d",
            name="New task",
            origin="new_task",
            contents=("new_task/output",),
        )
    )

    session.log_events([blueprint])

    assert recording_backend.events == [blueprint]


def test_session_begins_replacement_frame_and_forwards_timestamp(
    recording_backend: RecordingBackend,
) -> None:
    session = VisualizationSession(recording_backend)

    session.begin_frame(70, timestamp=12.5)

    assert recording_backend.steps == [70]
    assert recording_backend.timestamps == [12.5]
    assert recording_backend.paths_of(ClearEvent) == [
        "scene/lidar",
        "scene/prediction",
        "scene/ground_truth",
        "scene/cameras",
        "scene/meta",
    ]
    assert all(event.recursive for event in recording_backend.events)


def test_session_from_config_builds_the_configured_backend() -> None:
    session = VisualizationSession.from_config(VisualizationSessionConfig(backend="noop"))

    session.set_step(0)
    session.log_detection3d(_EMPTY_DETECTION)


def test_session_logs_detection_ground_truth(
    recording_backend: RecordingBackend,
) -> None:
    session = VisualizationSession(recording_backend)

    session.log_detection3d_data(
        points=_TWO_POINTS,
        gt_boxes=np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        gt_labels=np.array([1], dtype=np.int64),
        class_names=["pedestrian", "car"],
    )

    assert recording_backend.paths_of(Boxes3DEvent) == ["scene/ground_truth/detections"]


def test_session_logs_segmentation_predictions(
    recording_backend: RecordingBackend,
) -> None:
    session = VisualizationSession(recording_backend)

    session.log_segmentation3d(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        gt_labels=np.array([1, 1], dtype=np.int64),
        class_names=["road", "car"],
    )

    point_paths = recording_backend.paths_of(PointCloud3DEvent)
    assert "scene/prediction/segmentation" in point_paths
    assert "scene/ground_truth/segmentation" in point_paths
    assert "scene/lidar/intensity" in point_paths


def test_session_logs_segmentation_data(recording_backend: RecordingBackend) -> None:
    session = VisualizationSession(recording_backend)

    session.log_segmentation3d_data(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        class_names=["road", "car"],
    )

    assert "scene/ground_truth/segmentation" in recording_backend.paths_of(PointCloud3DEvent)


def test_session_logs_calibration_status(
    recording_backend: RecordingBackend, preview_calibration_data: CalibrationData
) -> None:
    session = VisualizationSession(recording_backend)

    session.log_calibration_status(preview_calibration_data, sample_name="sample-1")

    assert recording_backend.paths_of(Transform3DEvent) == ["calibration_status/camera"]


def test_session_logs_multiview_cameras(
    recording_backend: RecordingBackend, tmp_path: Path
) -> None:
    image_path = tmp_path / "cam.png"
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    camera = {
        "img_path": str(image_path),
        "cam2img": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 18.0], [0.0, 0.0, 1.0]],
        "lidar2cam": np.eye(4, dtype=np.float32).tolist(),
    }
    session = VisualizationSession(recording_backend)

    session.log_cameras({"CAM_FRONT": camera, "CAM_BACK": camera})

    assert recording_backend.paths_of(ImageEvent) == [
        "scene/cameras/CAM_FRONT",
        "scene/cameras/CAM_BACK",
    ]


def test_session_logs_persistent_camera_projection_layers(
    recording_backend: RecordingBackend, tmp_path: Path
) -> None:
    image_path = tmp_path / "cam.png"
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    camera = {
        "img_path": str(image_path),
        "cam2img": [[10.0, 0.0, 32.0], [0.0, 10.0, 18.0], [0.0, 0.0, 1.0]],
        "lidar2cam": np.eye(4, dtype=np.float32),
    }
    session = VisualizationSession(recording_backend)

    session.log_cameras(
        {"CAM_FRONT": camera},
        point_layers={
            "prediction/segmentation": CameraPointProjection(
                points=np.array([[0.0, 0.0, 10.0]], dtype=np.float32),
                labels=np.array([0], dtype=np.int64),
            )
        },
    )

    assert recording_backend.paths_of(Points2DEvent) == [
        "scene/cameras/CAM_FRONT/projected/prediction/segmentation"
    ]


def test_session_logs_the_scene_extent_only_when_there_is_geometry(
    recording_backend: RecordingBackend,
) -> None:
    session = VisualizationSession(recording_backend)

    session.log_scene_extent(
        np.zeros((2, 3), dtype=np.float32),
        boxes=(np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32),),
    )
    session.log_scene_extent(None)

    assert recording_backend.paths_of(PointCloud3DEvent) == ["scene/meta/extent"]
    assert len(recording_backend.events) == 1
