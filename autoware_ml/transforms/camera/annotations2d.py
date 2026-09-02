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
    """Compute the 8 corners of gravity-center 3D boxes.

    A numpy sibling of :meth:`autoware_ml.geometry.bbox_3d.LiDARBBox3D.corners`
    (which is torch-based) for use inside numpy transform pipelines.

    Args:
        boxes: Boxes ``(N, >=7)`` as ``[x, y, z_center, dx, dy, dz, yaw, ...]``,
            where ``z_center`` is the gravity center — the convention this
            repo's 3D boxes carry throughout.

    Returns:
        Corners with shape ``(N, 8, 3)``.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
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


def _project_points(
    points: np.ndarray, lidar2cam: np.ndarray, cam2img: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project lidar-frame points into image pixels.

    Returns:
        Pixel coordinates ``(N, 2)`` and an in-front-of-camera mask ``(N,)``.
    """
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_cam = (lidar2cam @ homogeneous.T).T
    valid = points_cam[:, 2] > 0
    projected = (cam2img[:3, :3] @ points_cam[:, :3].T).T
    projected = projected / np.maximum(projected[:, 2:3], 1e-6)
    return projected[:, :2], valid


class LoadAnnotations2DFromBoxes3D(BaseTransform):
    """Project 3D ground-truth boxes onto every camera as 2D annotations.

    Outputs (per camera, one entry per visible box):
        ``gt_bboxes_2d``: ``(N, 4)`` clipped ``(x1, y1, x2, y2)`` pixel boxes.
        ``centers_2d``: ``(N, 2)`` projected gravity centers in pixels.
        ``gt_labels_2d``: ``(N,)`` class labels.
    """

    _required_keys = ["img", "gt_boxes", "gt_labels", "lidar2cam", "camera_intrinsics"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute per-camera 2D annotations from the 3D boxes."""
        image_list, _ = as_hwc_image_list(input_dict["img"])
        image_height, image_width = int(image_list[0].shape[0]), int(image_list[0].shape[1])
        gt_boxes = np.asarray(input_dict["gt_boxes"], dtype=np.float32)
        if gt_boxes.size == 0:
            gt_boxes = gt_boxes.reshape(0, 9)
        if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 7:
            raise ValueError(
                f"gt_boxes must be (N, >=7) [x, y, z, dx, dy, dz, yaw, ...], "
                f"got shape {gt_boxes.shape}."
            )
        gt_labels = np.asarray(input_dict["gt_labels"]).reshape(-1)
        corners = _boxes3d_corners(gt_boxes)
        gravity_centers = gt_boxes[:, :3].copy()

        all_bboxes, all_centers, all_labels = [], [], []
        num_cams = len(input_dict["lidar2cam"])
        for camera_index in range(num_cams):
            lidar2cam = np.asarray(input_dict["lidar2cam"][camera_index], dtype=np.float64)
            cam2img = np.asarray(input_dict["camera_intrinsics"][camera_index], dtype=np.float64)
            bboxes, centers, labels = self._project_camera(
                corners,
                gravity_centers,
                gt_labels,
                lidar2cam,
                cam2img,
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

    def _project_camera(
        self,
        corners: np.ndarray,
        gravity_centers: np.ndarray,
        gt_labels: np.ndarray,
        lidar2cam: np.ndarray,
        cam2img: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_boxes = corners.shape[0]
        if num_boxes == 0:
            return self._empty_annotations()

        points = np.concatenate([corners.reshape(-1, 3), gravity_centers], axis=0)
        pixels, in_front = _project_points(points, lidar2cam, cam2img)
        corner_pixels = pixels[: num_boxes * 8].reshape(num_boxes, 8, 2)
        corner_front = in_front[: num_boxes * 8].reshape(num_boxes, 8)
        center_pixels = pixels[num_boxes * 8 :]
        center_front = in_front[num_boxes * 8 :]

        # A box qualifies when its gravity center projects inside this camera's
        # image (matching the reference recipe, which drops boxes whose center
        # leaves the crop — a clamped center could land outside its own box)
        # and at least one corner is in front of the camera. Corners behind
        # the camera are excluded from the 2D extent: their clamped-depth
        # projection is meaningless.
        candidates = np.nonzero(
            center_front
            & corner_front.any(axis=1)
            & (center_pixels[:, 0] >= 0)
            & (center_pixels[:, 0] < image_width)
            & (center_pixels[:, 1] >= 0)
            & (center_pixels[:, 1] < image_height)
        )[0]
        if candidates.size == 0:
            return self._empty_annotations()

        masked = np.where(corner_front[candidates, :, None], corner_pixels[candidates], np.nan)
        x_min = np.clip(np.nanmin(masked[:, :, 0], axis=1), 0, image_width)
        x_max = np.clip(np.nanmax(masked[:, :, 0], axis=1), 0, image_width)
        y_min = np.clip(np.nanmin(masked[:, :, 1], axis=1), 0, image_height)
        y_max = np.clip(np.nanmax(masked[:, :, 1], axis=1), 0, image_height)
        visible = (x_min < x_max) & (y_min < y_max)
        if not visible.any():
            return self._empty_annotations()
        kept = candidates[visible]

        bboxes = np.stack(
            [x_min[visible], y_min[visible], x_max[visible], y_max[visible]], axis=1
        ).astype(np.float32)
        centers = center_pixels[kept].astype(np.float32)
        return bboxes, centers, gt_labels[kept].astype(np.int64)

    @staticmethod
    def _empty_annotations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
