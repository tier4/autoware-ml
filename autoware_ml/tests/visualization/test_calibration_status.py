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

import numpy as np

from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus
from autoware_ml.visualization.calibration_status import build_calibration_status_events
from autoware_ml.visualization.events import (
    ImageEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextEvent,
    Transform3DEvent,
)

_IN_FRONT_OF_CAMERA = np.array([[0.0, 0.0, 10.0, 0.4]], dtype=np.float32)
_FUSED_IMAGE = np.zeros((720, 1280, 5), dtype=np.float32)
_FUSED_IMAGE[100, 100, 3] = 0.5
_FUSED_IMAGE[200, 200, 4] = 0.8


def test_build_calibration_status_events_includes_frames_and_status(
    preview_calibration_data: CalibrationData,
) -> None:
    events = build_calibration_status_events(
        preview_calibration_data,
        points=_IN_FRONT_OF_CAMERA,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        fused_image=_FUSED_IMAGE,
        gt_status=CalibrationStatus.CALIBRATED.value,
        pred_status=CalibrationStatus.MISCALIBRATED.value,
        pred_score=0.9,
        sample_name="sample-1",
    )
    paths = {event.path for event in events}

    assert "calibration_status/camera" in paths
    assert "calibration_status/lidar/points" in paths
    assert "calibration_status/status/gt_label" in paths
    assert "calibration_status/status/pred_label" in paths
    assert "calibration_status/camera/fused" in paths
    assert "calibration_status/camera/fused/depth" in paths
    assert "calibration_status/camera/fused/intensity" in paths
    fused_overlays = {
        event.path: event
        for event in events
        if isinstance(event, Points2DEvent)
        and event.path.startswith("calibration_status/camera/fused/")
    }
    assert fused_overlays["calibration_status/camera/fused/depth"].positions.shape == (1, 2)
    assert fused_overlays["calibration_status/camera/fused/intensity"].positions.shape == (1, 2)
    assert any(
        isinstance(event, Points2DEvent)
        and event.path == "calibration_status/camera/image/projected_points"
        for event in events
    )
    assert any(
        isinstance(event, ScalarEvent) and event.path == "calibration_status/status/pred_score"
        for event in events
    )


def test_build_calibration_status_events_summarizes_prediction_and_ground_truth(
    preview_calibration_data: CalibrationData,
) -> None:
    events = build_calibration_status_events(
        preview_calibration_data,
        gt_status=CalibrationStatus.CALIBRATED.value,
        pred_status=CalibrationStatus.MISCALIBRATED.value,
        pred_score=0.9,
    )

    summary = next(
        event
        for event in events
        if isinstance(event, TextEvent) and event.path == "calibration_status/status/summary"
    )
    assert summary.text == "pred: miscalibrated (0.90) | gt: calibrated"


def test_build_calibration_status_events_logs_camera_geometry_with_an_image(
    preview_calibration_data: CalibrationData,
) -> None:
    events = build_calibration_status_events(
        preview_calibration_data,
        image=np.zeros((36, 64, 3), dtype=np.uint8),
    )

    pinhole = next(event for event in events if isinstance(event, PinholeEvent))
    assert pinhole.resolution == (64, 36)
    assert any(
        isinstance(event, ImageEvent) and event.path == "calibration_status/camera/image"
        for event in events
    )
    assert any(isinstance(event, Transform3DEvent) for event in events)


def test_build_calibration_status_events_needs_only_calibration_data(
    preview_calibration_data: CalibrationData,
) -> None:
    """Without points or images the adapter still logs the camera frame."""
    events = build_calibration_status_events(preview_calibration_data)

    assert [event.path for event in events] == ["calibration_status/camera"]
    assert isinstance(events[0], Transform3DEvent)


def test_build_calibration_status_events_skips_projection_behind_the_camera(
    preview_calibration_data: CalibrationData,
) -> None:
    """Points with non-positive depth produce no image overlay."""
    events = build_calibration_status_events(
        preview_calibration_data,
        points=np.array([[0.0, 0.0, -50.0, 0.4]], dtype=np.float32),
        image=np.zeros((36, 64, 3), dtype=np.uint8),
    )

    assert not any(isinstance(event, Points2DEvent) for event in events)
    assert any(
        isinstance(event, PointCloud3DEvent) and event.path == "calibration_status/lidar/points"
        for event in events
    )


def test_build_calibration_status_events_honors_the_root_path(
    preview_calibration_data: CalibrationData,
) -> None:
    events = build_calibration_status_events(
        preview_calibration_data,
        sample_name="sample-1",
        root_path="dataset/calibration_status",
    )

    assert all(event.path.startswith("dataset/calibration_status") for event in events)
