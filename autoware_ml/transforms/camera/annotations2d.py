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

"""Derive per-camera 2D annotations from 3D boxes for auxiliary supervision.

The transform projects the (already augmented) 3D ground-truth boxes onto
every camera and emits per-camera 2D boxes, projected gravity centers, and
labels. It must therefore run after all geometric augmentations
(``ResizeCropFlipRotImage``, ``GlobalRotScaleTrans``, ``PadMultiViewImage``)
so the projection matrices and the pixels agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import as_hwc_image_list


def _boxes3d_corners(boxes: np.ndarray) -> np.ndarray:
    """Compute the 8 corners ``(N, 8, 3)`` of gravity-center boxes ``(N, >=7)``.

    A numpy sibling of the torch-based
    :attr:`autoware_ml.geometry.bbox_3d.lidar_bbox3d.LidarBBoxes3D.corners`.
    """
    dims = boxes[:, 3:6]
    signs = np.array(
        [[dx, dy, dz] for dx in (-0.5, 0.5) for dy in (-0.5, 0.5) for dz in (-0.5, 0.5)],
        dtype=np.float32,
    )
    corners = dims[:, None, :] * signs[None, :, :]
    yaw = boxes[:, 6]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotated_x = corners[..., 0] * cos_yaw[:, None] - corners[..., 1] * sin_yaw[:, None]
    rotated_y = corners[..., 0] * sin_yaw[:, None] + corners[..., 1] * cos_yaw[:, None]
    corners = np.stack([rotated_x, rotated_y, corners[..., 2]], axis=-1)
    return corners + boxes[:, None, :3]


def _project_points(points: np.ndarray, lidar2img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project lidar-frame points to pixels ``(N, 2)`` and an in-front-of-camera mask ``(N,)``."""
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    projected = homogeneous @ lidar2img.T
    depth = projected[:, 2:3]
    return projected[:, :2] / np.maximum(depth, 1e-6), depth[:, 0] > 0


def _empty_annotations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def _project_camera(
    corners: np.ndarray,
    gravity_centers: np.ndarray,
    gt_labels: np.ndarray,
    lidar2img: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_boxes = corners.shape[0]
    if num_boxes == 0:
        return _empty_annotations()

    points = np.concatenate([corners.reshape(-1, 3), gravity_centers], axis=0)
    pixels, in_front = _project_points(points, lidar2img)
    corner_pixels = pixels[: num_boxes * 8].reshape(num_boxes, 8, 2)
    corner_front = in_front[: num_boxes * 8].reshape(num_boxes, 8)
    center_pixels = pixels[num_boxes * 8 :]
    center_front = in_front[num_boxes * 8 :]

    # The center must project inside the image (a clamped center could land
    # outside its own clipped box) and at least one corner must be in front.
    candidates = np.nonzero(
        center_front
        & corner_front.any(axis=1)
        & (center_pixels[:, 0] >= 0)
        & (center_pixels[:, 0] < image_width)
        & (center_pixels[:, 1] >= 0)
        & (center_pixels[:, 1] < image_height)
    )[0]
    if candidates.size == 0:
        return _empty_annotations()

    # Corners behind the camera have no meaningful projection; leave them out.
    masked = np.where(corner_front[candidates, :, None], corner_pixels[candidates], np.nan)
    x_min = np.clip(np.nanmin(masked[:, :, 0], axis=1), 0, image_width)
    x_max = np.clip(np.nanmax(masked[:, :, 0], axis=1), 0, image_width)
    y_min = np.clip(np.nanmin(masked[:, :, 1], axis=1), 0, image_height)
    y_max = np.clip(np.nanmax(masked[:, :, 1], axis=1), 0, image_height)
    visible = (x_min < x_max) & (y_min < y_max)
    if not visible.any():
        return _empty_annotations()
    kept = candidates[visible]

    bboxes = np.stack([x_min[visible], y_min[visible], x_max[visible], y_max[visible]], axis=1)
    return (
        bboxes.astype(np.float32),
        center_pixels[kept].astype(np.float32),
        gt_labels[kept].astype(np.int64),
    )


class LoadAnnotations2DFromBoxes3D(BaseTransform):
    """Project 3D ground-truth boxes onto every camera as 2D annotations.

    Outputs (per camera, one entry per visible box):
        ``gt_bboxes_2d``: ``(N, 4)`` clipped ``(x1, y1, x2, y2)`` pixel boxes.
        ``centers_2d``: ``(N, 2)`` projected gravity centers in pixels.
        ``gt_labels_2d``: ``(N,)`` class labels.
    """

    _required_keys = ["img", "gt_boxes", "gt_labels", "lidar2img"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute per-camera 2D annotations from the 3D boxes."""
        image_list, _ = as_hwc_image_list(input_dict["img"])
        image_height, image_width = image_list[0].shape[:2]
        gt_boxes = np.asarray(input_dict["gt_boxes"], dtype=np.float32)
        gt_labels = np.asarray(input_dict["gt_labels"]).reshape(-1)
        corners = _boxes3d_corners(gt_boxes)
        gravity_centers = gt_boxes[:, :3]

        all_bboxes, all_centers, all_labels = [], [], []
        for lidar2img in input_dict["lidar2img"]:
            bboxes, centers, labels = _project_camera(
                corners,
                gravity_centers,
                gt_labels,
                np.asarray(lidar2img, dtype=np.float64),
                image_height,
                image_width,
            )
            all_bboxes.append(bboxes)
            all_centers.append(centers)
            all_labels.append(labels)

        return {
            "gt_bboxes_2d": all_bboxes,
            "centers_2d": all_centers,
            "gt_labels_2d": all_labels,
        }
