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
import numpy.typing as npt
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.transforms.base import BaseTransform


class UndistortImage(BaseTransform):
    """Undistort every image of the sample and clear its distortion coefficients.

    Undistortion is not an affine map of the pixels, so it is not composed into the image
    augmentation matrices. It replaces the intrinsics of the camera, which every later
    transform and the view transform then read.
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
        """Undistort every image of the sample.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with undistorted images.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        images = []
        camera_intrinsics = []
        for index, image in enumerate(camera_image_data.images):
            coefficients = camera_image_data.distortion_coefficients[index].numpy()
            intrinsic = camera_image_data.augmented_camera_intrinsics[index].numpy()
            if not coefficients.size:
                images.append(image)
                camera_intrinsics.append(camera_image_data.augmented_camera_intrinsics[index])
                continue

            undistorted, new_intrinsic = self.undistort(
                np.transpose(image.numpy(), (1, 2, 0)), intrinsic, coefficients
            )
            images.append(torch.from_numpy(np.transpose(undistorted, (2, 0, 1)).copy()))
            camera_intrinsics.append(torch.from_numpy(new_intrinsic))

        augmented_camera_intrinsics = torch.stack(camera_intrinsics)
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(
                    update={
                        "images": torch.stack(images),
                        "augmented_camera_intrinsics": augmented_camera_intrinsics,
                        "distortion_coefficients": [
                            torch.zeros(0, dtype=torch.float32)
                            for _ in camera_image_data.camera_names
                        ],
                        "distortion_models": ["" for _ in camera_image_data.camera_names],
                        "lidar2images": compose_lidar2images(
                            augmented_camera_intrinsics, camera_image_data.lidar2cams
                        ),
                    }
                )
            )
        )

    def undistort(
        self,
        image: npt.NDArray[np.float32],
        camera_intrinsic: npt.NDArray[np.float32],
        distortion_coefficients: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Undistort one image and return it with the intrinsics it is expressed in.

        Args:
            image: Image in (height, width, num_channels) layout.
            camera_intrinsic: 3x3 intrinsics of the camera.
            distortion_coefficients: Distortion coefficients of the camera.

        Returns:
            tuple: The undistorted image and the new 3x3 intrinsics.
        """
        height, width = image.shape[:2]
        new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
            camera_intrinsic,
            distortion_coefficients,
            (width, height),
            self.alpha,
            (width, height),
        )
        undistorted = cv2.undistort(
            image, camera_intrinsic, distortion_coefficients, newCameraMatrix=new_intrinsic
        )
        return undistorted, new_intrinsic.astype(np.float32)
