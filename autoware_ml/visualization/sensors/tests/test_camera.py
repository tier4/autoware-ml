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
import tempfile
import unittest

import cv2
from jaxtyping import Float32
import numpy as np

from autoware_ml.visualization.sensors.camera import (
    CameraBoxProjection,
    CameraCalibration,
    CameraPointProjection,
    build_camera_events,
)
from autoware_ml.visualization.viewers.events import (
    ImageEvent,
    LineStrips2DEvent,
    PinholeEvent,
    Points2DEvent,
    Transform3DEvent,
)


def _intrinsics(focal: float, principal_point: tuple[float, float]) -> Float32[np.ndarray, "3 3"]:
    """Return pinhole intrinsics with one focal length and a principal point."""
    centre_x, centre_y = principal_point
    return np.array(
        [[focal, 0.0, centre_x], [0.0, focal, centre_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _offset_extrinsic() -> Float32[np.ndarray, "4 4"]:
    """Return an identity-rotation LiDAR-to-camera transform translated by (1, 2, 3)."""
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, 3] = (1.0, 2.0, 3.0)
    return extrinsic


def _camera(
    image_path: Path,
    *,
    intrinsics: Float32[np.ndarray, "3 3"] | None = None,
    lidar_to_camera: Float32[np.ndarray, "4 4"] | None = None,
) -> CameraCalibration:
    """Build one camera calibration, defaulting to the offset extrinsic."""
    return CameraCalibration(
        image_path=str(image_path),
        intrinsics=_intrinsics(1000.0, (640.0, 360.0)) if intrinsics is None else intrinsics,
        lidar_to_camera=_offset_extrinsic() if lidar_to_camera is None else lidar_to_camera,
    )


def _centred_camera(image_path: Path, focal: float) -> CameraCalibration:
    """Build a camera at the LiDAR origin whose principal point is the image centre."""
    return _camera(
        image_path,
        intrinsics=_intrinsics(focal, (32.0, 18.0)),
        lidar_to_camera=np.eye(4, dtype=np.float32),
    )


class _CameraImageTestCase(unittest.TestCase):
    """Base case that writes one small BGR camera image to a temporary directory."""

    def setUp(self) -> None:
        """Write one 64x36 black image to disk."""
        self.tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.image_path = self.tmp_path / "cam_front.png"
        cv2.imwrite(str(self.image_path), np.zeros((36, 64, 3), dtype=np.uint8))


class TestCameraEntities(_CameraImageTestCase):
    """Every camera is logged as a transform, pinhole and image triplet."""

    def test_emits_one_triplet_per_camera(self) -> None:
        """Each camera gets one transform, one pinhole and one image event under its path."""
        events = build_camera_events(
            {
                "CAM_FRONT": _camera(self.image_path),
                "CAM_BACK": _camera(self.image_path),
            }
        )

        expected_paths = ["scene/cameras/CAM_FRONT", "scene/cameras/CAM_BACK"]
        for event_type in (Transform3DEvent, PinholeEvent, ImageEvent):
            with self.subTest(event_type=event_type.__name__):
                self.assertEqual(
                    [event.path for event in events if isinstance(event, event_type)],
                    expected_paths,
                )

    def test_reads_resolution_from_the_image(self) -> None:
        """The pinhole resolution is the width and height of the image on disk."""
        events = build_camera_events({"CAM_FRONT": _camera(self.image_path)})

        pinhole = next(event for event in events if isinstance(event, PinholeEvent))
        self.assertEqual(pinhole.resolution, (64, 36))

    def test_splits_the_extrinsic_into_rotation_and_translation(self) -> None:
        """The LiDAR-to-camera transform becomes a translation and a rotation matrix."""
        events = build_camera_events({"CAM_FRONT": _camera(self.image_path)})

        transform = next(event for event in events if isinstance(event, Transform3DEvent))
        np.testing.assert_allclose(transform.translation, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(transform.rotation_matrix, np.eye(3, dtype=np.float32))

    def test_honors_the_root_path(self) -> None:
        """Every camera entity is logged under the requested root path."""
        events = build_camera_events({"CAM_FRONT": _camera(self.image_path)}, root_path="multiview")

        self.assertTrue(all(event.path == "multiview/CAM_FRONT" for event in events))

    def test_is_empty_without_cameras(self) -> None:
        """A sample without cameras produces no events."""
        self.assertEqual(list(build_camera_events({})), [])


class TestCameraErrors(_CameraImageTestCase):
    """Invalid camera inputs fail loudly instead of silently dropping data."""

    def test_raises_on_unreadable_image(self) -> None:
        """A missing image must fail loudly rather than silently drop the camera."""
        camera = _camera(self.tmp_path / "does_not_exist.png")

        with self.assertRaisesRegex(FileNotFoundError, "does_not_exist.png"):
            build_camera_events({"CAM_FRONT": camera})

    def test_rejects_misaligned_overlay_data(self) -> None:
        """Point labels that do not align with the points are rejected."""
        with self.assertRaisesRegex(ValueError, "labels must align with points"):
            build_camera_events(
                {"CAM_FRONT": _camera(self.image_path)},
                point_layers={
                    "prediction/segmentation": CameraPointProjection(
                        points=np.zeros((2, 3), dtype=np.float32),
                        labels=np.zeros((1,), dtype=np.int64),
                    )
                },
            )

    def test_rejects_non_positive_projection_limit(self) -> None:
        """A projection limit of zero is rejected."""
        with self.assertRaisesRegex(ValueError, "max_projected_points"):
            build_camera_events(
                {"CAM_FRONT": _camera(self.image_path)},
                max_projected_points=0,
            )


class TestProjectedLayers(_CameraImageTestCase):
    """Point and box layers are projected persistently onto every camera image."""

    def test_projects_semantic_points_into_every_camera(self) -> None:
        """Labelled points in front of the camera land on their pixels with their classes."""
        camera = _centred_camera(self.image_path, focal=10.0)
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
        self.assertEqual(
            [event.path for event in projections],
            [
                "scene/cameras/CAM_FRONT/projected/prediction/segmentation",
                "scene/cameras/CAM_BACK/projected/prediction/segmentation",
            ],
        )
        np.testing.assert_allclose(projections[0].positions, [[32.0, 18.0], [33.0, 18.0]])
        np.testing.assert_array_equal(projections[0].class_ids, [0, 1])
        self.assertIsNotNone(projections[0].colors)

    def test_projects_ground_truth_and_prediction_boxes(self) -> None:
        """Each box layer becomes twelve wireframe edges coloured by class."""
        box = np.array([[0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32)
        events = build_camera_events(
            {"CAM_FRONT": _centred_camera(self.image_path, focal=10.0)},
            box_layers={
                "ground_truth/detections": CameraBoxProjection(box, np.array([0], dtype=np.int64)),
                "prediction/detections": CameraBoxProjection(box, np.array([1], dtype=np.int64)),
            },
        )

        projections = [event for event in events if isinstance(event, LineStrips2DEvent)]
        self.assertEqual(
            [event.path for event in projections],
            [
                "scene/cameras/CAM_FRONT/projected/ground_truth/detections",
                "scene/cameras/CAM_FRONT/projected/prediction/detections",
            ],
        )
        self.assertEqual(len(projections[0].strips), 12)
        self.assertIsNotNone(projections[0].colors)
        np.testing.assert_array_equal(projections[1].class_ids, np.ones((12,), dtype=np.int64))

    def test_filters_and_caps_projected_points(self) -> None:
        """Points outside the image are dropped and the rest are capped to the limit."""
        points = np.column_stack(
            (
                np.linspace(-10.0, 10.0, 100),
                np.zeros((100,)),
                np.full((100,), 10.0),
            )
        ).astype(np.float32)

        events = build_camera_events(
            {"CAM_FRONT": _centred_camera(self.image_path, focal=1.0)},
            point_layers={"lidar": CameraPointProjection(points)},
            max_projected_points=7,
        )

        projection = next(event for event in events if isinstance(event, Points2DEvent))
        self.assertEqual(projection.positions.shape, (7, 2))
        self.assertIsNotNone(projection.colors)
        self.assertEqual(projection.colors.shape, (7, 4))


if __name__ == "__main__":
    unittest.main()
