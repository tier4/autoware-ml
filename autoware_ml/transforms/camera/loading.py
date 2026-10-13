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

"""Camera loading transforms to support ModelGTSample."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


class LoadImagesFromFile(BaseTransform):
    """Load the images of every camera of the sample together with their calibration."""

    _required_keys = ["image_samples"]

    def __init__(self) -> None:
        """Initialize the image loader, which serves the pixel values in [0, 1]."""
        super().__init__(probability=None)

    def load_image(self, image_path: str) -> torch.Tensor:
        """Read one RGB image from disk as a (num_channels, height, width) tensor.

        Args:
            image_path: Path of the image file.

        Returns:
            torch.Tensor: The image in channel first layout.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(np.transpose(image, (2, 0, 1)).copy())

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Load the images of the sample and their calibration.

        Args:
            model_gt_sample: ModelGTSample instance containing `image_samples`.

        Returns:
            Updated ModelGTSample instance with loaded `camera_image_data`.
        """
        # This is checked in the _validate_required_keys()
        image_samples = model_gt_sample.image_samples  # type: ignore[reportOptionalIterable]
        if not image_samples:
            raise ValueError("No image samples found in the ModelGTSample.")

        images = torch.stack([self.load_image(sample.image_path) for sample in image_samples])
        camera_intrinsics = torch.stack([sample.camera_intrinsic for sample in image_samples])
        camera_image_data = BaseImages(
            images=images,
            timestamps=torch.tensor(
                [sample.timestamp for sample in image_samples], dtype=torch.float64
            ),
            camera_intrinsics=camera_intrinsics,
            camera_names=[sample.camera_name for sample in image_samples],
            lidar2images=torch.stack([sample.lidar2image for sample in image_samples]),
            lidar2cams=torch.stack([sample.lidar2cam for sample in image_samples]),
            distortion_models=[sample.distortion_model for sample in image_samples],
            distortion_coefficients=[sample.distortion_coefficients for sample in image_samples],
            augmented_camera_intrinsics=camera_intrinsics.clone(),
            image_augmentation_matrices=BaseImages.identity_image_augmentation_matrices(
                camera_intrinsics
            ),
        )
        return model_gt_sample._replace(camera_image_data=camera_image_data)
