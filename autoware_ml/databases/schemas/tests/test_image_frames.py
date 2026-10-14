"""Tests for the image frame schema."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.databases.schemas.image_frames import (
    ImageFrameDataModel,
    distortion_model_from_coefficients,
)


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([], ""),
        ([0.1, -0.2, 0.001, 0.002], "plumb_bob"),
        ([0.1, -0.2, 0.001, 0.002, 0.05], "plumb_bob"),
        ([0.1, -0.2, 0.001, 0.002, 0.05, 0.0, 0.0, 0.0], "rational_polynomial"),
        ([0.0] * 14, "rational_polynomial"),
    ],
)
def test_distortion_model_from_coefficients(coefficients: list[float], expected: str) -> None:
    assert distortion_model_from_coefficients(coefficients) == expected


def test_image_frame_round_trips_distortion_through_dictionary() -> None:
    frame = ImageFrameDataModel(
        image_frame_id="frame",
        image_keyframe=True,
        image_sensor_id="sensor",
        image_sensor_channel_name="CAM_FRONT",
        image_timestamp_seconds=1.0,
        image_path="a.jpg",
        image_height=10,
        image_width=20,
        cam2img=np.eye(3, dtype=np.float64),
        image_distortion_coefficients=[0.1, -0.2, 0.001, 0.002, 0.05],
        image_distortion_model="plumb_bob",
        image_sensor_to_ego_pose_matrix=np.eye(4, dtype=np.float64),
        image_frame_ego_pose_to_global_matrix=np.eye(4, dtype=np.float64),
        lidar2cam=None,
        lidar2img=None,
    )

    restored = ImageFrameDataModel.load_from_dictionary(frame.to_dictionary())

    assert list(restored.image_distortion_coefficients) == [0.1, -0.2, 0.001, 0.002, 0.05]
    assert restored.image_distortion_model == "plumb_bob"
