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

import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


def _project_camera(
    corners: torch.Tensor,
    gravity_centers: torch.Tensor,
    gt_labels: torch.Tensor,
    lidar2image: torch.Tensor,
    image_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the boxes onto one camera, following the reference StreamPETR recipe.

    A box is kept when at least one of its corners is in front of the camera and its
    extent, clipped to the image, has a non-zero area; the corners behind the camera have
    no meaningful projection and are left out of the extent. The projected gravity center
    is clamped into the image, so a box whose center leaves the image is still a target.

    Args:
        corners: Box corners ``(N, 8, 3)`` in the lidar frame.
        gravity_centers: Box centers ``(N, 3)`` in the lidar frame.
        gt_labels: Class index of every box ``(N,)``.
        lidar2image: Lidar-to-image projection ``(4, 4)`` of the camera.
        image_size: ``[width, height]`` of the served images, in pixels.

    Returns:
        The clipped ``(x1, y1, x2, y2)`` boxes, the clamped centers and the labels of the
        kept boxes.
    """
    num_boxes = corners.shape[0]
    points = torch.cat([corners.reshape(-1, 3), gravity_centers], dim=0)
    projected = torch.cat([points, points.new_ones(points.shape[0], 1)], dim=1) @ lidar2image.T
    pixels = projected[:, :2] / projected[:, 2:3]
    in_front = projected[:, 2] > 0
    corner_pixels = pixels[: num_boxes * 8].view(num_boxes, 8, 2)
    corner_front = in_front[: num_boxes * 8].view(num_boxes, 8, 1)

    low = torch.where(corner_front, corner_pixels, torch.inf).amin(dim=1).clamp(min=0.0)
    high = torch.where(corner_front, corner_pixels, -torch.inf).amax(dim=1).clamp(min=0.0)
    low, high = torch.minimum(low, image_size), torch.minimum(high, image_size)
    centers = torch.minimum(pixels[num_boxes * 8 :].clamp(min=0.0), image_size)
    keep = corner_front.any(dim=1).squeeze(1) & (low < high).all(dim=1)
    return torch.cat([low[keep], high[keep]], dim=1), centers[keep], gt_labels[keep]


class LoadAnnotations2DFromBoxes3D(BaseTransform):
    """Project the 3D ground-truth boxes onto every camera as 2D annotations.

    Every camera gets one entry per visible box: the clipped ``(x1, y1, x2, y2)``
    pixel box, the projected gravity center clamped into the image and the class
    label, exactly the targets of the reference StreamPETR recipe. The box
    centers are gravity centers, the only center the 3D boxes take.
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
        image_size = torch.tensor([image_width, image_height], dtype=torch.float32)

        all_bboxes, all_centers, all_labels = [], [], []
        for lidar2image in camera_image_data.lidar2images:
            camera_bboxes, camera_centers, camera_labels = _project_camera(
                bboxes.corners, bboxes.center, bboxes.bbox_labels, lidar2image, image_size
            )
            all_bboxes.append(camera_bboxes)
            all_centers.append(camera_centers)
            all_labels.append(camera_labels)

        return model_gt_sample._replace(
            detection2d_gt_sample=Detection2DGTSample(
                gt_bboxes_2d=all_bboxes, gt_centers_2d=all_centers, gt_labels_2d=all_labels
            )
        )
