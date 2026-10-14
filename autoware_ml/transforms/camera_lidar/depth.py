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

"""Camera-lidar depth transforms.

Projects the lidar points of a sample onto every camera image to build sparse depth maps,
used e.g. as depth guidance by the LSS view transforms.
"""

from __future__ import annotations

from jaxtyping import Float32
import torch
from torch import Tensor

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


class LiDARDepthSparseTransform(BaseTransform):
    """Project the lidar points onto every camera into sparse depth maps.

    The depth maps are stored in ``camera_image_data.depth_images`` with the resolution of the
    images. A pixel hit by a lidar point holds the depth of that point along the camera's
    optical axis, every other pixel is zero.

    ``lidar2images`` already contains the image-space augmentations applied earlier in the
    pipeline, as the loading and image transforms keep it in sync with ``images``, so the
    points are projected with it directly and an identity image augmentation matrix. Hence
    this transform must run after every transform that changes the image geometry.

    Required keys:
        - ``point_cloud_data``: points with at least XYZ columns.
        - ``camera_image_data``: images and their ``lidar2images`` projections.

    Optional keys:
        - None

    Generated keys:
        - ``camera_image_data.depth_images``: sparse depth maps of shape
          ``(num_cameras, height, width)``.
    """

    _required_keys = ["point_cloud_data", "camera_image_data"]

    def __init__(self) -> None:
        """Initialize the LiDARDepthSparseTransform transform."""
        super().__init__(probability=None)

    def build_depth_maps(
        self,
        points: Float32[Tensor, "num_points num_point_features"],
        lidar2image: Float32[Tensor, "num_cameras 4 4"],
        image_size: tuple[int, int],
    ) -> Float32[Tensor, "num_cameras height width"]:
        """Project the lidar points of one sample onto each camera into sparse depth maps.

        Both matrices must map into the same image plane the depth map is
        built for: when ``lidar2image`` already contains the image
        augmentation (training pipeline), pass an identity ``img_aug_matrix``;
        at deployment the runtime provides the raw projection and the
        augmentation separately.

        The per-camera scatter is vectorized so the traced graph supports a
        dynamic number of cameras; duplicate pixel hits resolve to an
        arbitrary point among the duplicates.

        Args:
            points: Lidar points with at least XYZ columns.
            lidar2image: Lidar-to-image projection of every camera, shape ``(num_cameras, 4, 4)``.
            image_size: Height and width of the depth maps, i.e. of the images.

        Returns:
            Sparse depth maps of shape ``(num_cameras, height, width)``.
        """
        num_cams = lidar2image.shape[0]
        height, width = image_size

        coords = points[:, :3].transpose(0, 1)  # (3, P)
        # Rotation projection points into the image plane of every camera.
        # (num_cameras, 3, 3) @ (3, P) = (num_cameras, 3, P)
        projected = lidar2image[:, :3, :3].matmul(coords)
        # Translation moves the projected points into the image plane of every camera.
        # (num_cameras, 3, P) + (num_cameras, 3, 1) = (num_cameras, 3, P)
        projected = projected + lidar2image[:, :3, 3].reshape(-1, 3, 1)

        distances = projected[:, 2, :]  # (num_cameras, P)
        valid_distance = distances > 0

        projected = torch.cat(
            [
                projected[:, :2, :] / torch.clamp(projected[:, 2:3, :], 1e-5, 1e5),
                projected[:, 2:3, :],
            ],
            dim=1,
        )
        pixel_coords = projected[:, :2, :].transpose(1, 2)[..., [1, 0]]  # (num_cameras, P, [y, x])

        on_image = (
            (pixel_coords[..., 0] >= 0)
            & (pixel_coords[..., 0] < height)
            & (pixel_coords[..., 1] >= 0)
            & (pixel_coords[..., 1] < width)
            & valid_distance
        )
        hits = torch.nonzero(on_image, as_tuple=False)
        camera_indices = hits[:, 0]
        point_indices = hits[:, 1]
        hit_coords = pixel_coords[camera_indices, point_indices].long()
        hit_distances = distances[camera_indices, point_indices]

        flat_indices = camera_indices * height * width + hit_coords[:, 0] * width + hit_coords[:, 1]
        flat_depth = lidar2image.new_zeros(num_cams * height * width)
        flat_depth.scatter_(dim=0, index=flat_indices, src=hit_distances)
        return flat_depth.view(num_cams, height, width)

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        """Build the sparse depth maps of every camera from the lidar points.

        Args:
            multi_task_gt_sample: ModelGTSample instance containing `point_cloud_data` and
                `camera_image_data`.

        Returns:
            Updated ModelGTSample instance with `depth_images` set in `camera_image_data`.
        """
        assert multi_task_gt_sample.point_cloud_data is not None
        assert multi_task_gt_sample.camera_image_data is not None
        point_cloud_data = multi_task_gt_sample.point_cloud_data
        camera_image_data = multi_task_gt_sample.camera_image_data

        lidar2images = camera_image_data.lidar2images
        height, width = camera_image_data.images.shape[-2:]

        depth_images = self.build_depth_maps(
            points=point_cloud_data.points.to(lidar2images),
            # lidar2images already maps into the augmented image plane, so it doesn't need
            # image augmentation.
            lidar2image=lidar2images,
            image_size=(int(height), int(width)),
        )
        # Expand to (num_cameras, 1, height, width) to match the expected shape of depth maps.
        depth_images = depth_images.unsqueeze(1)

        # Only the depth changes, every other field is carried over unchanged.
        # model_copy does not validate what it is given, so the copy is validated explicitly.
        camera_image_data_with_depth = BaseImages.model_validate(
            camera_image_data.model_copy(update={"depth_maps": depth_images})
        )
        return multi_task_gt_sample._replace(camera_image_data=camera_image_data_with_depth)
