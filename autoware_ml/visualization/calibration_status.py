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

"""Calibration-status visualization adapters."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus
from autoware_ml.visualization.colors import depths_to_colors, scalar_to_heatmap_colors
from autoware_ml.visualization.common import (
    as_numpy,
    build_sample_metadata_events,
    ensure_image_uint8,
    ensure_xyz,
)
from autoware_ml.visualization.events import (
    ImageEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextEvent,
    Transform3DEvent,
    VisualizationEvent,
)

_STATUS_TEXT = {
    CalibrationStatus.MISCALIBRATED.value: "miscalibrated",
    CalibrationStatus.CALIBRATED.value: "calibrated",
}


def _fused_channel_points(fused_image: Any, channel: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract nonzero normalized BGRDI pixels as a 2D heatmap point overlay."""
    array = as_numpy(fused_image)
    if array.ndim != 3:
        raise ValueError(f"fused_image must have shape (H, W, 5) or (5, H, W), got {array.shape}")
    if array.shape[0] <= 8 and array.shape[1] > 8 and array.shape[2] > 8:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[2] < 5:
        raise ValueError(f"fused_image must have five channels, got {array.shape}")
    values = np.clip(array[:, :, channel], 0.0, 1.0).astype(np.float32, copy=False)
    valid = values > 0.0
    rows, columns = np.nonzero(valid)
    positions = np.column_stack((columns, rows)).astype(np.float32, copy=False)
    colors = scalar_to_heatmap_colors(values[valid], alpha=220)
    return positions, colors


def _project_points_to_image(
    points: np.ndarray,
    calibration_data: CalibrationData,
    image_shape: tuple[int, int],
    *,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lidar points into the current image plane."""
    point_positions = ensure_xyz(points)
    if point_positions.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    points_hom = np.concatenate(
        [point_positions, np.ones((point_positions.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_camera = (
        calibration_data.lidar_to_camera_transformation.astype(np.float32, copy=False)
        @ points_hom.T
    ).T[:, :3]
    valid_depth_mask = points_camera[:, 2] > 0.0
    if not np.any(valid_depth_mask):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    points_camera = points_camera[valid_depth_mask]
    point_depths = points_camera[:, 2].astype(np.float32, copy=False)
    projected_points, _ = cv2.projectPoints(
        points_camera,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        calibration_data.new_camera_matrix.astype(np.float32, copy=False),
        calibration_data.distortion_coefficients.astype(np.float32, copy=False),
    )
    projected_points = projected_points.reshape(-1, 2).astype(np.float32, copy=False)

    height, width = image_shape
    in_frame_mask = (
        (projected_points[:, 0] >= 0.0)
        & (projected_points[:, 0] < width)
        & (projected_points[:, 1] >= 0.0)
        & (projected_points[:, 1] < height)
    )
    projected_points = projected_points[in_frame_mask]
    point_depths = point_depths[in_frame_mask]

    if projected_points.shape[0] > max_points:
        keep_indices = np.linspace(0, projected_points.shape[0] - 1, num=max_points, dtype=np.int64)
        projected_points = projected_points[keep_indices]
        point_depths = point_depths[keep_indices]

    return projected_points, point_depths


def _build_status_summary_text(
    gt_status: int | None,
    pred_status: int | None,
    pred_score: float | None,
) -> str | None:
    """Build one readable calibration summary."""
    summary_parts: list[str] = []
    if pred_status is not None:
        pred_label = _STATUS_TEXT.get(pred_status, str(pred_status))
        if pred_score is None:
            summary_parts.append(f"pred: {pred_label}")
        else:
            summary_parts.append(f"pred: {pred_label} ({pred_score:.2f})")
    if gt_status is not None:
        summary_parts.append(f"gt: {_STATUS_TEXT.get(gt_status, str(gt_status))}")
    if not summary_parts:
        return None
    return " | ".join(summary_parts)


def build_calibration_status_events(
    calibration_data: CalibrationData,
    *,
    points: np.ndarray | None = None,
    image: np.ndarray | None = None,
    fused_image: np.ndarray | None = None,
    gt_status: int | None = None,
    pred_status: int | None = None,
    pred_score: float | None = None,
    sample_name: str | None = None,
    root_path: str = "calibration_status",
) -> list[VisualizationEvent]:
    """Build backend-neutral calibration visualization events for one sample."""
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)

    transform = calibration_data.lidar_to_camera_transformation.astype(np.float32, copy=False)
    events.append(
        Transform3DEvent(
            path=f"{root_path}/camera",
            translation=transform[:3, 3],
            rotation_matrix=transform[:3, :3],
        )
    )

    if image is not None:
        image_uint8 = ensure_image_uint8(image)
        events.append(
            PinholeEvent(
                path=f"{root_path}/camera",
                image_from_camera=calibration_data.new_camera_matrix.astype(np.float32, copy=False),
                resolution=(int(image_uint8.shape[1]), int(image_uint8.shape[0])),
            )
        )
        events.append(ImageEvent(f"{root_path}/camera/image", image_uint8))

        if points is not None:
            projected_points, point_depths = _project_points_to_image(
                points,
                calibration_data,
                image_uint8.shape[:2],
            )
            if projected_points.shape[0] > 0:
                overlay_colors = depths_to_colors(point_depths)
                events.append(
                    Points2DEvent(
                        path=f"{root_path}/camera/image/projected_points",
                        positions=projected_points,
                        colors=overlay_colors,
                        radii=2.0,
                    )
                )
                events.append(
                    ScalarEvent(
                        f"{root_path}/metrics/num_projected_points",
                        float(projected_points.shape[0]),
                    )
                )

    if fused_image is not None:
        fused_path = f"{root_path}/camera/fused"
        events.append(ImageEvent(fused_path, ensure_image_uint8(fused_image)))
        depth_positions, depth_colors = _fused_channel_points(fused_image, 3)
        if depth_positions.shape[0] > 0:
            events.append(
                Points2DEvent(
                    f"{fused_path}/depth",
                    positions=depth_positions,
                    colors=depth_colors,
                    radii=2.0,
                )
            )
        intensity_positions, intensity_colors = _fused_channel_points(fused_image, 4)
        if intensity_positions.shape[0] > 0:
            events.append(
                Points2DEvent(
                    f"{fused_path}/intensity",
                    positions=intensity_positions,
                    colors=intensity_colors,
                    radii=2.0,
                )
            )

    if points is not None:
        point_positions = ensure_xyz(points)
        events.append(PointCloud3DEvent(f"{root_path}/lidar/points", point_positions))

    if gt_status is not None:
        events.append(ScalarEvent(f"{root_path}/status/gt", float(gt_status)))
        events.append(
            TextEvent(
                f"{root_path}/status/gt_label",
                _STATUS_TEXT.get(gt_status, str(gt_status)),
            )
        )

    if pred_status is not None:
        events.append(ScalarEvent(f"{root_path}/status/pred", float(pred_status)))
        events.append(
            TextEvent(
                f"{root_path}/status/pred_label",
                _STATUS_TEXT.get(pred_status, str(pred_status)),
            )
        )
    if pred_score is not None:
        events.append(ScalarEvent(f"{root_path}/status/pred_score", float(pred_score)))

    summary_text = _build_status_summary_text(gt_status, pred_status, pred_score)
    if summary_text is not None:
        events.append(TextEvent(f"{root_path}/status/summary", summary_text))

    return events
