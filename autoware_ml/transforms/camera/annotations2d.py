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

"""Per-camera 2D annotations projected from the 3D boxes, for auxiliary 2D supervision.

The projection reads the calibration and the image size of the sample as they are when it
runs, so the transform must run after every geometric augmentation, the image-space ones and
the global 3D ones alike. It is a pinhole projection through ``lidar2images`` that ignores the
distortion coefficients, as the reference recipe does, so the images must be undistorted.
"""

from __future__ import annotations

import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.geometry import Box2DFieldIndex


def _project_camera(
    corners: torch.Tensor,
    gravity_centers: torch.Tensor,
    gt_labels: torch.Tensor,
    lidar2image: torch.Tensor,
    image_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the boxes onto one camera by the rule of the reference recipe.

    The rule is that of the reference's ``compute_bbox_and_centers``: a box is kept when at
    least one of its corners is in front of the camera and its extent, clipped to the image,
    has a non-zero area; the corners behind the camera have no meaningful projection and are
    left out of the extent. The projected gravity center is clamped into the image, so a box
    whose center leaves the image is still a target. A center behind the camera is divided by
    its negative depth and clamped all the same, as in the reference.

    Args:
        corners: Box corners ``(N, 8, 3)`` in the lidar frame.
        gravity_centers: Box centers ``(N, 3)`` in the lidar frame.
        gt_labels: Class index of every box ``(N,)``.
        lidar2image: Lidar-to-image projection ``(4, 4)`` of the camera.
        image_size: ``[width, height]`` of the served images, in pixels.

    Returns:
        The kept boxes, one row per box laid out by Box2DFieldIndex: the clipped x_min,
        y_min, x_max, y_max, then the clamped center_x, center_y of the projected gravity
        center; and the labels of the kept boxes.
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
    columns = {
        Box2DFieldIndex.X_MIN: low[:, 0],
        Box2DFieldIndex.Y_MIN: low[:, 1],
        Box2DFieldIndex.X_MAX: high[:, 0],
        Box2DFieldIndex.Y_MAX: high[:, 1],
        Box2DFieldIndex.CENTER_X: centers[:, 0],
        Box2DFieldIndex.CENTER_Y: centers[:, 1],
    }
    boxes = torch.stack([columns[field] for field in Box2DFieldIndex], dim=1)
    return boxes[keep], gt_labels[keep]


class LoadAnnotations2DFromBoxes3D(BaseTransform):
    """Project the 3D ground-truth boxes onto every camera as 2D annotations.

    The sample's Detection2DGTSample gets, for every camera, one Box2DFieldIndex row and one
    label per box the camera sees, by the rule of ``_project_camera``. The center columns hold
    the projected gravity center, the only center the 3D boxes take.
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

        corners, centers, labels = bboxes.corners, bboxes.center, bboxes.bbox_labels
        all_bboxes, all_labels = [], []
        for lidar2image in camera_image_data.lidar2images:
            camera_bboxes, camera_labels = _project_camera(
                corners, centers, labels, lidar2image, image_size
            )
            all_bboxes.append(camera_bboxes)
            all_labels.append(camera_labels)

        return model_gt_sample._replace(
            detection2d_gt_sample=Detection2DGTSample(
                gt_bboxes_2d=all_bboxes, gt_labels_2d=all_labels
            )
        )
