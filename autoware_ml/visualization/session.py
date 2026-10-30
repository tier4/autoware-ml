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

"""High-level visualization session facade."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from jaxtyping import Float32, Int64, Shaped
import numpy as np

from autoware_ml.visualization.common import build_scene_extent_event
from autoware_ml.visualization.entity_paths import (
    CALIBRATION_STATUS_PATH,
    CAMERAS_PATH,
    DATA_CALIBRATION_STATUS_PATH,
    GROUND_TRUTH_PATH,
    LIDAR_PATH,
    META_PATH,
    PREDICTION_PATH,
    SCENE_PATH,
)
from autoware_ml.visualization.sensors.camera import (
    CameraBoxProjection,
    CameraCalibration,
    CameraPointProjection,
    build_camera_events,
)
from autoware_ml.visualization.tasks.calibration_status import build_calibration_status_events
from autoware_ml.visualization.tasks.detection3d import (
    build_detection3d_data_events,
    build_detection3d_events,
)
from autoware_ml.visualization.tasks.segmentation3d import (
    build_segmentation3d_data_events,
    build_segmentation3d_events,
)
from autoware_ml.visualization.viewers.backends import create_visualization_backend
from autoware_ml.visualization.viewers.contracts import (
    VisualizationBackend,
    VisualizationSessionConfig,
)
from autoware_ml.visualization.viewers.events import ClearEvent, VisualizationEvent

#: Entities that hold the geometry of one frame and are cleared before the next one: the scene
#: layers, and the image, LiDAR and sample name of a calibration-status sample under both of its
#: roots. Calibration overlays are only logged when they hold points, so without the clear an
#: empty overlay would leave the previous frame's points over the new image. The metric and status
#: scalars are kept for their plots.
FRAME_ENTITY_PATHS = (
    LIDAR_PATH,
    PREDICTION_PATH,
    GROUND_TRUTH_PATH,
    CAMERAS_PATH,
    META_PATH,
    *(
        f"{root}/{entity}"
        for root in (CALIBRATION_STATUS_PATH, DATA_CALIBRATION_STATUS_PATH)
        for entity in ("camera", "lidar", "meta")
    ),
)


class VisualizationSession:
    """Wrap one backend and expose task-oriented logging helpers."""

    def __init__(self, backend: VisualizationBackend) -> None:
        """
        Initialize the visualization session.

        Args:
            backend: Backend every event is logged to.
        """
        self.backend = backend

    @classmethod
    def from_config(cls, config: VisualizationSessionConfig) -> VisualizationSession:
        """
        Construct a session with the backend its configuration names.

        Args:
            config: Session configuration.

        Returns:
            VisualizationSession: The session.
        """
        return cls(create_visualization_backend(config))

    def set_step(self, step: int) -> None:
        """
        Advance the visualization timeline.

        Args:
            step: Timeline step of the next events.
        """
        self.backend.set_step(step)

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """
        Log events from an external task adapter through the neutral API.

        Args:
            events: Events to log.
        """
        self.backend.log_events(events)

    def begin_frame(self, step: int, *, timestamp: float | None = None) -> None:
        """Start a replacement-style scene frame and remove stale geometry.

        Rerun resolves entities using latest-at semantics. Explicit clears are
        therefore required when a 10 Hz prediction frame has no corresponding
        1 Hz ground truth; otherwise the previous GT would remain visible.
        Clearing LiDAR and prediction paths also guarantees that point clouds
        from earlier frames are never accumulated as pseudo multi-sweeps.
        """
        self.backend.set_step(step)
        if timestamp is not None:
            self.backend.set_timestamp(timestamp)
        self.backend.log_events([ClearEvent(path) for path in FRAME_ENTITY_PATHS])

    def log_scene_extent(
        self,
        points: Float32[np.ndarray, "num_points num_features"] | None,
        *,
        boxes: Sequence[Float32[np.ndarray, "num_boxes box_params"]],
        root_path: str = SCENE_PATH,
    ) -> None:
        """
        Log the invisible frame extent that keeps sibling 3D views aligned.

        Args:
            points: Points of the frame, None when it has none.
            boxes: Every box set the frame draws.
            root_path: Entity path of the scene.
        """
        event = build_scene_extent_event(points, boxes, root_path=root_path)
        if event is not None:
            self.backend.log_events([event])

    def log_calibration_status(
        self,
        *,
        lidar_to_camera: Float32[np.ndarray, "4 4"],
        image_from_camera: Float32[np.ndarray, "3 3"],
        points: Float32[np.ndarray, "num_points num_features"] | None = None,
        image: Shaped[np.ndarray, "dim_0 dim_1 dim_2"] | None = None,
        fused_image: Float32[np.ndarray, "dim_0 dim_1 dim_2"] | None = None,
        gt_status: int | None = None,
        pred_status: int | None = None,
        pred_score: float | None = None,
        sample_name: str | None = None,
        root_path: str = CALIBRATION_STATUS_PATH,
    ) -> None:
        """
        Log one calibration-status sample.

        See :func:`build_calibration_status_events` for the arguments.
        """
        self.backend.log_events(
            build_calibration_status_events(
                lidar_to_camera=lidar_to_camera,
                image_from_camera=image_from_camera,
                points=points,
                image=image,
                fused_image=fused_image,
                gt_status=gt_status,
                pred_status=pred_status,
                pred_score=pred_score,
                sample_name=sample_name,
                root_path=root_path,
            )
        )

    def log_segmentation3d(
        self,
        points: Float32[np.ndarray, "num_points num_features"],
        pred_labels: Int64[np.ndarray, " num_points"],
        *,
        pred_probs: Float32[np.ndarray, "num_points num_classes"] | None = None,
        gt_labels: Int64[np.ndarray, " num_points"] | None = None,
        class_names: Sequence[str] | None = None,
        ignore_index: int | None = None,
        root_path: str = SCENE_PATH,
        point_radius: float = 0.04,
        point_labels: bool = False,
        sample_name: str | None = None,
    ) -> None:
        """
        Log one 3D segmentation sample with predictions.

        See :func:`build_segmentation3d_events` for the arguments.
        """
        self.backend.log_events(
            build_segmentation3d_events(
                points,
                pred_labels,
                pred_probs=pred_probs,
                gt_labels=gt_labels,
                class_names=class_names,
                ignore_index=ignore_index,
                root_path=root_path,
                point_radius=point_radius,
                point_labels=point_labels,
                sample_name=sample_name,
            )
        )

    def log_segmentation3d_data(
        self,
        points: Float32[np.ndarray, "num_points num_features"],
        labels: Int64[np.ndarray, " num_points"],
        *,
        class_names: Sequence[str] | None = None,
        ignore_index: int | None = None,
        root_path: str = SCENE_PATH,
        point_radius: float = 0.04,
        point_labels: bool = False,
        sample_name: str | None = None,
    ) -> None:
        """
        Log one 3D segmentation sample without predictions.

        See :func:`build_segmentation3d_data_events` for the arguments.
        """
        self.backend.log_events(
            build_segmentation3d_data_events(
                points,
                labels,
                class_names=class_names,
                ignore_index=ignore_index,
                root_path=root_path,
                point_radius=point_radius,
                point_labels=point_labels,
                sample_name=sample_name,
            )
        )

    def log_detection3d(
        self,
        *,
        pred_boxes: Float32[np.ndarray, "num_predictions box_params"],
        pred_scores: Float32[np.ndarray, " num_predictions"],
        pred_labels: Int64[np.ndarray, " num_predictions"],
        points: Float32[np.ndarray, "num_points num_features"] | None,
        gt_boxes: Float32[np.ndarray, "num_boxes box_params"] | None = None,
        gt_labels: Int64[np.ndarray, " num_boxes"] | None = None,
        class_names: Sequence[str] | None = None,
        root_path: str = SCENE_PATH,
        point_radius: float = 0.04,
        sample_name: str | None = None,
    ) -> None:
        """
        Log one 3D detection sample with predictions.

        See :func:`build_detection3d_events` for the arguments.
        """
        self.backend.log_events(
            build_detection3d_events(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                root_path=root_path,
                point_radius=point_radius,
                sample_name=sample_name,
            )
        )

    def log_detection3d_data(
        self,
        *,
        points: Float32[np.ndarray, "num_points num_features"] | None,
        gt_boxes: Float32[np.ndarray, "num_boxes box_params"],
        gt_labels: Int64[np.ndarray, " num_boxes"],
        class_names: Sequence[str] | None = None,
        root_path: str = SCENE_PATH,
        point_radius: float = 0.04,
        sample_name: str | None = None,
    ) -> None:
        """
        Log one 3D detection sample without predictions.

        See :func:`build_detection3d_data_events` for the arguments.
        """
        self.backend.log_events(
            build_detection3d_data_events(
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                root_path=root_path,
                point_radius=point_radius,
                sample_name=sample_name,
            )
        )

    def log_cameras(
        self,
        cameras: Mapping[str, CameraCalibration],
        *,
        root_path: str = CAMERAS_PATH,
        point_layers: Mapping[str, CameraPointProjection] | None = None,
        box_layers: Mapping[str, CameraBoxProjection] | None = None,
        max_projected_points: int = 10_000,
    ) -> None:
        """
        Log camera images and their persistent projected layers.

        See :func:`build_camera_events` for the arguments.
        """
        self.backend.log_events(
            build_camera_events(
                cameras,
                root_path=root_path,
                point_layers=point_layers,
                box_layers=box_layers,
                max_projected_points=max_projected_points,
            )
        )
