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

"""Tests for the multiview camera visualization adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from autoware_ml.visualization.cameras import (
    CameraBoxProjection,
    CameraPointProjection,
    build_camera_events,
)
from autoware_ml.visualization.events import (
    ImageEvent,
    LineStrips2DEvent,
    PinholeEvent,
    Points2DEvent,
    Transform3DEvent,
)

_INTRINSIC = [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
_EXTRINSIC = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
]


@pytest.fixture
def camera_image(tmp_path: Path) -> Path:
    """Write one small BGR image to disk and return its path."""
    image_path = tmp_path / "cam_front.png"
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    return image_path


def _camera_entry(image_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build one camera info entry with optional overrides."""
    entry: dict[str, Any] = {
        "img_path": str(image_path),
        "cam2img": _INTRINSIC,
        "lidar2cam": _EXTRINSIC,
    }
    entry.update(overrides)
    return entry


def test_build_camera_events_emits_one_triplet_per_camera(camera_image: Path) -> None:
    events = build_camera_events(
        {
            "CAM_FRONT": _camera_entry(camera_image),
            "CAM_BACK": _camera_entry(camera_image),
        }
    )

    assert [event.path for event in events if isinstance(event, Transform3DEvent)] == [
        "scene/cameras/CAM_FRONT",
        "scene/cameras/CAM_BACK",
    ]
    assert [event.path for event in events if isinstance(event, PinholeEvent)] == [
        "scene/cameras/CAM_FRONT",
        "scene/cameras/CAM_BACK",
    ]
    assert [event.path for event in events if isinstance(event, ImageEvent)] == [
        "scene/cameras/CAM_FRONT",
        "scene/cameras/CAM_BACK",
    ]


def test_build_camera_events_reads_resolution_from_the_image(
    camera_image: Path,
) -> None:
    events = build_camera_events({"CAM_FRONT": _camera_entry(camera_image)})

    pinhole = next(event for event in events if isinstance(event, PinholeEvent))
    assert pinhole.resolution == (64, 36)


def test_build_camera_events_splits_the_extrinsic_into_rotation_and_translation(
    camera_image: Path,
) -> None:
    events = build_camera_events({"CAM_FRONT": _camera_entry(camera_image)})

    transform = next(event for event in events if isinstance(event, Transform3DEvent))
    np.testing.assert_allclose(transform.translation, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform.rotation_matrix, np.eye(3, dtype=np.float32))


def test_build_camera_events_honors_the_root_path(camera_image: Path) -> None:
    events = build_camera_events({"CAM_FRONT": _camera_entry(camera_image)}, root_path="multiview")

    assert all(event.path == "multiview/CAM_FRONT" for event in events)


@pytest.mark.parametrize("missing_key", ["img_path", "cam2img", "lidar2cam"])
def test_build_camera_events_raises_on_missing_calibration(
    camera_image: Path, missing_key: str
) -> None:
    """A camera missing calibration must fail loudly instead of being skipped."""
    entry = _camera_entry(camera_image, **{missing_key: None})

    with pytest.raises(ValueError, match=f"CAM_FRONT.*{missing_key}"):
        build_camera_events({"CAM_FRONT": entry})


def test_build_camera_events_raises_on_unreadable_image(tmp_path: Path) -> None:
    """A missing image must fail loudly rather than silently drop the camera."""
    entry = _camera_entry(tmp_path / "does_not_exist.png")

    with pytest.raises(FileNotFoundError, match="does_not_exist.png"):
        build_camera_events({"CAM_FRONT": entry})


def test_build_camera_events_is_empty_without_cameras() -> None:
    assert build_camera_events({}) == []


def test_build_camera_events_projects_semantic_points_into_every_camera(
    camera_image: Path,
) -> None:
    camera = _camera_entry(
        camera_image,
        cam2img=[[10.0, 0.0, 32.0], [0.0, 10.0, 18.0], [0.0, 0.0, 1.0]],
        lidar2cam=np.eye(4, dtype=np.float32),
    )
    events = build_camera_events(
        {"CAM_FRONT": camera, "CAM_BACK": camera},
        point_layers={
            "prediction/segmentation": CameraPointProjection(
                points=np.array(
                    [[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 0.0, -1.0]],
                    dtype=np.float32,
                ),
                labels=np.array([0, 1, 1], dtype=np.int64),
                class_names=["road", "car"],
            )
        },
    )

    projections = [event for event in events if isinstance(event, Points2DEvent)]
    assert [event.path for event in projections] == [
        "scene/cameras/CAM_FRONT/projected/prediction/segmentation",
        "scene/cameras/CAM_BACK/projected/prediction/segmentation",
    ]
    np.testing.assert_allclose(projections[0].positions, [[32.0, 18.0], [33.0, 18.0]])
    np.testing.assert_array_equal(projections[0].class_ids, [0, 1])
    assert projections[0].colors is not None


def test_build_camera_events_projects_ground_truth_and_prediction_boxes(
    camera_image: Path,
) -> None:
    camera = _camera_entry(
        camera_image,
        cam2img=[[10.0, 0.0, 32.0], [0.0, 10.0, 18.0], [0.0, 0.0, 1.0]],
        lidar2cam=np.eye(4, dtype=np.float32),
    )
    box = np.array([[0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    events = build_camera_events(
        {"CAM_FRONT": camera},
        box_layers={
            "ground_truth/detections": CameraBoxProjection(box, np.array([0])),
            "prediction/detections": CameraBoxProjection(box, np.array([1])),
        },
    )

    projections = [event for event in events if isinstance(event, LineStrips2DEvent)]
    assert [event.path for event in projections] == [
        "scene/cameras/CAM_FRONT/projected/ground_truth/detections",
        "scene/cameras/CAM_FRONT/projected/prediction/detections",
    ]
    assert len(projections[0].strips) == 12
    assert projections[0].colors is not None
    np.testing.assert_array_equal(projections[1].class_ids, np.ones((12,), dtype=np.int64))


def test_build_camera_events_filters_and_caps_projected_points(
    camera_image: Path,
) -> None:
    camera = _camera_entry(
        camera_image,
        cam2img=[[1.0, 0.0, 32.0], [0.0, 1.0, 18.0], [0.0, 0.0, 1.0]],
        lidar2cam=np.eye(4, dtype=np.float32),
    )
    points = np.column_stack(
        (
            np.linspace(-10.0, 10.0, 100),
            np.zeros((100,)),
            np.full((100,), 10.0),
        )
    ).astype(np.float32)

    events = build_camera_events(
        {"CAM_FRONT": camera},
        point_layers={"lidar": CameraPointProjection(points)},
        max_projected_points=7,
    )

    projection = next(event for event in events if isinstance(event, Points2DEvent))
    assert projection.positions.shape == (7, 2)
    assert projection.colors is not None
    assert projection.colors.shape == (7, 4)


def test_build_camera_events_rejects_misaligned_overlay_data(
    camera_image: Path,
) -> None:
    camera = _camera_entry(camera_image)

    with pytest.raises(ValueError, match="labels must align with points"):
        build_camera_events(
            {"CAM_FRONT": camera},
            point_layers={
                "prediction/segmentation": CameraPointProjection(
                    points=np.zeros((2, 3), dtype=np.float32),
                    labels=np.zeros((1,), dtype=np.int64),
                )
            },
        )


def test_build_camera_events_rejects_non_positive_projection_limit(
    camera_image: Path,
) -> None:
    with pytest.raises(ValueError, match="max_projected_points"):
        build_camera_events(
            {"CAM_FRONT": _camera_entry(camera_image)},
            max_projected_points=0,
        )
