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

"""Camera undistortion transforms to support ModelGTSample."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class UndistortImage(BaseTransform):
    """Undistort every camera image using its distortion coefficients.

    The distortion coefficients are expressed in the frame of the raw image, so this
    transform must run before any other image-space transform. Undistorted cameras have
    their entry in `augmented_camera_intrinsics` replaced by the optimal new camera matrix
    and their distortion coefficients zeroed, so applying the transform twice is a no-op.
    """

    _required_keys = ["camera_image_data"]

    def __init__(self, alpha: float = 0.0) -> None:
        """Initialize the UndistortImage transform.

        Args:
            alpha: Free scaling parameter passed to OpenCV undistortion. 0.0 crops invalid
                pixels, while 1.0 retains the full field of view.
        """
        super().__init__(probability=None)
        self.alpha = alpha

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Undistort every camera image and update its intrinsics.

        Args:
            model_gt_sample: ModelGTSample instance containing `camera_image_data`.

        Returns:
            Updated ModelGTSample instance with an undistorted `camera_image_data`.
        """
        assert model_gt_sample.camera_image_data is not None
        camera_image_data = model_gt_sample.camera_image_data

        images = camera_image_data.images
        camera_intrinsics = camera_image_data.camera_intrinsics
        augmented_camera_intrinsics = camera_image_data.augmented_camera_intrinsics.clone()

        undistorted_images = []
        undistorted_coefficients = []
        for index, coefficients in enumerate(camera_image_data.distortion_coefficients):
            # Nothing to correct for pre-undistorted cameras, keep the image as it is.
            if coefficients.numel() == 0 or not torch.any(coefficients):
                undistorted_images.append(images[index])
                undistorted_coefficients.append(coefficients)
                continue

            # OpenCV expects (height, width, num_channels) contiguous arrays.
            image = images[index].permute(1, 2, 0).contiguous().numpy()
            height, width = image.shape[:2]
            camera_matrix = camera_intrinsics[index].numpy().astype(np.float64)
            distortion_coefficients = coefficients.numpy().astype(np.float64)

            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                distortion_coefficients,
                (width, height),
                self.alpha,
                (width, height),
            )
            image = cv2.undistort(
                image,
                camera_matrix,
                distortion_coefficients,
                newCameraMatrix=new_camera_matrix,
            )

            undistorted_images.append(torch.from_numpy(image).permute(2, 0, 1).to(images.dtype))
            augmented_camera_intrinsics[index] = torch.from_numpy(new_camera_matrix).to(
                augmented_camera_intrinsics.dtype
            )
            undistorted_coefficients.append(torch.zeros(0, dtype=coefficients.dtype))

        # The undistortion is not a 2D affine, so it is carried by the intrinsics alone and
        # leaves the composed image augmentation matrices untouched. The projections are
        # rebuilt from the new intrinsics, and the coefficients are cleared so no later
        # projection distorts the points onto the now pinhole images.
        # model_copy does not validate what it is given, so the copy is validated explicitly.
        undistorted_camera_image_data = BaseImages.model_validate(
            camera_image_data.model_copy(
                update={
                    "images": torch.stack(undistorted_images, dim=0),
                    "distortion_coefficients": undistorted_coefficients,
                    "distortion_models": ["" for _ in camera_image_data.camera_names],
                    "augmented_camera_intrinsics": augmented_camera_intrinsics,
                    "lidar2images": compose_lidar2images(
                        augmented_camera_intrinsics, camera_image_data.lidar2cams
                    ),
                }
            )
        )
        return model_gt_sample._replace(camera_image_data=undistorted_camera_image_data)
