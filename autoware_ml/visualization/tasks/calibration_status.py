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

from collections.abc import Sequence
from jaxtyping import Float32, Shaped, UInt8
import numpy as np

from autoware_ml.utils.calibration import CalibrationStatus
from autoware_ml.visualization.colors import depths_to_colors, scalar_to_heatmap_colors
from autoware_ml.visualization.entity_paths import CALIBRATION_STATUS_PATH
from autoware_ml.visualization.common import (
    build_sample_metadata_events,
    ensure_image_uint8,
    ensure_xyz,
)
from autoware_ml.visualization.viewers.events import (
    ImageEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextDocumentEvent,
    TextEvent,
    Transform3DEvent,
    VisualizationEvent,
)

_STATUS_TEXT = {
    CalibrationStatus.MISCALIBRATED.value: "miscalibrated",
    CalibrationStatus.CALIBRATED.value: "calibrated",
}


def _fused_channel_points(
    fused_image: Float32[np.ndarray, "dim_0 dim_1 dim_2"], channel: int
) -> tuple[Float32[np.ndarray, "num_pixels 2"], UInt8[np.ndarray, "num_pixels 4"]]:
    """Extract nonzero normalized BGRDI pixels as a 2D heatmap point overlay."""
    array = np.asarray(fused_image)
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
    points: Float32[np.ndarray, "num_points num_features"],
    lidar_to_camera: Float32[np.ndarray, "4 4"],
    image_from_camera: Float32[np.ndarray, "3 3"],
    image_shape: tuple[int, int],
    *,
    max_points: int = 5000,
) -> tuple[Float32[np.ndarray, "num_projected 2"], Float32[np.ndarray, " num_projected"]]:
    """Project lidar points into the current image plane.

    The calibration pipelines undistort the images before anything else touches them, so a
    pinhole projection with the intrinsics of the shown image is exact.
    """
    point_positions = ensure_xyz(points)
    if point_positions.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    points_hom = np.concatenate(
        [point_positions, np.ones((point_positions.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_camera = (lidar_to_camera.astype(np.float32, copy=False) @ points_hom.T).T[:, :3]
    valid_depth_mask = points_camera[:, 2] > 0.0
    if not np.any(valid_depth_mask):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    points_camera = points_camera[valid_depth_mask]
    point_depths = points_camera[:, 2].astype(np.float32, copy=False)
    projected_points = (image_from_camera.astype(np.float32, copy=False) @ points_camera.T).T
    projected_points = (projected_points[:, :2] / projected_points[:, 2:3]).astype(
        np.float32, copy=False
    )

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


def _build_status_verdict_document(
    gt_status: int | None,
    pred_status: int | None,
    pred_score: float | None,
) -> str | None:
    """Build the markdown verdict shown in its own pane under the images.

    The log-stream summary is easy to miss, so the verdict repeats the
    prediction and the ground truth as headings and states whether they agree.
    """
    details: list[str] = []
    if pred_status is not None:
        pred_label = _STATUS_TEXT.get(pred_status, str(pred_status)).upper()
        score = "" if pred_score is None else f" ({pred_score:.2f})"
        details.append(f"Prediction: {pred_label}{score}")
    if gt_status is not None:
        details.append(f"Ground truth: {_STATUS_TEXT.get(gt_status, str(gt_status)).upper()}")
    if not details:
        return None
    # The agreement is the one word that matters, so it is the headline; the
    # two statuses share the line under it so both fit a short pane.
    if pred_status is not None and gt_status is not None:
        headline = "MATCH" if pred_status == gt_status else "MISMATCH"
        return f"# {headline}\n\n## " + " \u00b7 ".join(details)
    return "# " + details[0]


def build_calibration_status_events(
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
) -> Sequence[VisualizationEvent]:
    """Build backend-neutral calibration visualization events for one sample.

    Args:
        lidar_to_camera: Homogeneous lidar to camera transform the sample was fused with,
            including any misalignment the pipeline applied.
        image_from_camera: Intrinsics of the shown image, after every image-space transform.
        points: Lidar points of the sample.
        image: Camera image, HWC or CHW.
        fused_image: Image with the depth and intensity channels appended, HWC or CHW.
        gt_status: Ground-truth calibration status.
        pred_status: Predicted calibration status.
        pred_score: Probability of the predicted status.
        sample_name: Name logged with the sample.
        root_path: Entity path under which the events are logged.

    Returns:
        The events of the sample.
    """
    events: list[VisualizationEvent] = [*build_sample_metadata_events(root_path, sample_name)]

    transform = np.asarray(lidar_to_camera, dtype=np.float32)
    intrinsics = np.asarray(image_from_camera, dtype=np.float32)
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
                image_from_camera=intrinsics,
                resolution=(int(image_uint8.shape[1]), int(image_uint8.shape[0])),
            )
        )
        events.append(ImageEvent(f"{root_path}/camera/image", image_uint8))

        if points is not None:
            projected_points, point_depths = _project_points_to_image(
                points,
                transform,
                intrinsics,
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
    verdict_document = _build_status_verdict_document(gt_status, pred_status, pred_score)
    if verdict_document is not None:
        events.append(TextDocumentEvent(f"{root_path}/status/verdict", verdict_document))

    return events
