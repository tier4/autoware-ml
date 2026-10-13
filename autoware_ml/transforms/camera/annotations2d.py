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
labels. It must therefore run after every geometric augmentation (the image
augmentations and the global 3D ones), so the projection matrices and the
pixels agree.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


def _project_points(
    points: npt.NDArray, lidar2img: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Project lidar-frame points to pixels ``(N, 2)`` and an in-front-of-camera mask ``(N,)``."""
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    projected = homogeneous @ lidar2img.T
    depth = projected[:, 2:3]
    return projected[:, :2] / np.maximum(depth, 1e-6), depth[:, 0] > 0


def _empty_annotations() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def _project_camera(
    corners: npt.NDArray,
    gravity_centers: npt.NDArray,
    gt_labels: npt.NDArray,
    lidar2img: npt.NDArray[np.float64],
    image_height: int,
    image_width: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
    """Project the 3D ground-truth boxes onto every camera as 2D annotations.

    Every camera gets one entry per visible box: the clipped ``(x1, y1, x2, y2)``
    pixel box, the projected gravity center in pixels and the class label. The
    box centers are gravity centers, the only center the 3D boxes take.
    """

    _required_keys = ["camera_image_data", "detection3d_gt_bboxes_3d"]

    def __init__(self) -> None:
        """Initialize the transform, which projects every sample."""
        super().__init__(probability=None)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Compute the per-camera 2D annotations from the 3D boxes.

        Args:
            model_gt_sample: Sample holding the loaded images and the 3D boxes.

        Returns:
            The sample with its per-camera 2D annotations.
        """
        # These are checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )
        bboxes: BaseBBoxes3D = (
            model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        )
        image_height, image_width = camera_image_data.images.shape[-2:]
        corners = bboxes.corners.numpy()
        gravity_centers = bboxes.center.numpy()
        gt_labels = bboxes.bbox_labels.numpy()

        all_bboxes, all_centers, all_labels = [], [], []
        for lidar2image in camera_image_data.lidar2images:
            camera_bboxes, camera_centers, camera_labels = _project_camera(
                corners,
                gravity_centers,
                gt_labels,
                lidar2image.numpy().astype(np.float64),
                int(image_height),
                int(image_width),
            )
            all_bboxes.append(torch.from_numpy(camera_bboxes))
            all_centers.append(torch.from_numpy(camera_centers))
            all_labels.append(torch.from_numpy(camera_labels))

        return model_gt_sample._replace(
            detection2d_gt_sample=Detection2DGTSample(
                gt_bboxes_2d=all_bboxes, gt_centers_2d=all_centers, gt_labels_2d=all_labels
            )
        )
