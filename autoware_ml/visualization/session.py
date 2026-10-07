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
from typing import Any

from autoware_ml.utils.calibration import CalibrationData
from autoware_ml.visualization.backends import create_visualization_backend
from autoware_ml.visualization.calibration_status import build_calibration_status_events
from autoware_ml.visualization.cameras import (
    CameraBoxProjection,
    CameraPointProjection,
    build_camera_events,
)
from autoware_ml.visualization.contracts import (
    VisualizationBackend,
    VisualizationSessionConfig,
)
from autoware_ml.visualization.detection3d import (
    build_detection3d_data_events,
    build_detection3d_events,
)
from autoware_ml.visualization.common import build_scene_extent_event
from autoware_ml.visualization.events import ClearEvent, VisualizationEvent
from autoware_ml.visualization.segmentation3d import (
    build_segmentation3d_data_events,
    build_segmentation3d_events,
)


class VisualizationSession:
    """Wrap one backend and expose task-oriented logging helpers."""

    def __init__(self, backend: VisualizationBackend) -> None:
        """Initialize the visualization session from a backend instance."""
        self.backend = backend

    @classmethod
    def from_config(cls, config: VisualizationSessionConfig) -> VisualizationSession:
        """Construct a session from configuration."""
        return cls(create_visualization_backend(config))

    def set_step(self, step: int) -> None:
        """Advance the visualization timeline."""
        self.backend.set_step(step)

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Log events from an external task adapter through the neutral API."""
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
        self.backend.log_events(
            [
                ClearEvent("scene/lidar"),
                ClearEvent("scene/prediction"),
                ClearEvent("scene/ground_truth"),
                ClearEvent("scene/cameras"),
                ClearEvent("scene/meta"),
            ]
        )

    def log_scene_extent(
        self,
        points: Any | None,
        *,
        boxes: Sequence[Any | None] = (),
        root_path: str = "scene",
    ) -> None:
        """Log the invisible frame extent that keeps sibling 3D views aligned."""
        event = build_scene_extent_event(points, boxes, root_path=root_path)
        if event is not None:
            self.backend.log_events([event])

    def log_calibration_status(
        self,
        calibration_data: CalibrationData,
        *,
        points: Any | None = None,
        image: Any | None = None,
        fused_image: Any | None = None,
        gt_status: int | None = None,
        pred_status: int | None = None,
        pred_score: float | None = None,
        sample_name: str | None = None,
        root_path: str = "calibration_status",
    ) -> None:
        """Log one calibration-status sample."""
        self.backend.log_events(
            build_calibration_status_events(
                calibration_data,
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
        points: Any,
        pred_labels: Any,
        *,
        pred_probs: Any | None = None,
        gt_labels: Any | None = None,
        class_names: Sequence[str] | None = None,
        ignore_index: int | None = None,
        root_path: str = "scene",
        point_radius: float = 0.04,
        point_labels: bool = False,
        sample_name: str | None = None,
        point_color_mode: str = "semantic",
        pred_logits: Any | None = None,
    ) -> None:
        """Log one 3D segmentation sample."""
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
                point_color_mode=point_color_mode,
                pred_logits=pred_logits,
            )
        )

    def log_segmentation3d_data(
        self,
        points: Any,
        labels: Any,
        *,
        class_names: Sequence[str] | None = None,
        ignore_index: int | None = None,
        root_path: str = "scene",
        point_radius: float = 0.04,
        point_labels: bool = False,
        sample_name: str | None = None,
        point_color_mode: str = "semantic",
    ) -> None:
        """Log one transformed 3D segmentation sample without predictions."""
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
                point_color_mode=point_color_mode,
            )
        )

    def log_detection3d(
        self,
        predictions: Mapping[str, Any],
        *,
        points: Any | None = None,
        gt_boxes: Any | None = None,
        gt_labels: Any | None = None,
        class_names: Sequence[str] | None = None,
        root_path: str = "scene",
        point_radius: float = 0.04,
        sample_name: str | None = None,
        point_color_mode: str = "semantic",
    ) -> None:
        """Log one 3D detection sample."""
        self.backend.log_events(
            build_detection3d_events(
                predictions,
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                root_path=root_path,
                point_radius=point_radius,
                sample_name=sample_name,
                point_color_mode=point_color_mode,
            )
        )

    def log_cameras(
        self,
        images: dict[str, Any],
        *,
        root_path: str = "scene/cameras",
        point_layers: Mapping[str, CameraPointProjection] | None = None,
        box_layers: Mapping[str, CameraBoxProjection] | None = None,
        max_projected_points: int = 10_000,
    ) -> None:
        """Log camera images and persistent projected LiDAR/task overlays."""
        self.backend.log_events(
            build_camera_events(
                images,
                root_path=root_path,
                point_layers=point_layers,
                box_layers=box_layers,
                max_projected_points=max_projected_points,
            )
        )

    def log_detection3d_data(
        self,
        *,
        points: Any | None = None,
        gt_boxes: Any,
        gt_labels: Any,
        class_names: Sequence[str] | None = None,
        root_path: str = "scene",
        point_radius: float = 0.04,
        sample_name: str | None = None,
        point_color_mode: str = "semantic",
    ) -> None:
        """Log one transformed 3D detection sample without predictions."""
        self.backend.log_events(
            build_detection3d_data_events(
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                root_path=root_path,
                point_radius=point_radius,
                sample_name=sample_name,
                point_color_mode=point_color_mode,
            )
        )
