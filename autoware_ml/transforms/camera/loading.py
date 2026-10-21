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

import numpy as np
import torch
from torchvision.io import decode_image

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.geometry import ImageChannel


class LoadImagesFromFile(BaseTransform):
    """Load the images of every camera of the sample together with their calibration.

    The cameras are stacked along the leading dimension in the order of the image samples. The
    pixel values are served in unit range by default, which is the range the models and the
    camera-lidar fusion read.

    Required keys:
        - image_samples: one record per camera, with the image path and the calibration.

    Generated keys:
        - camera_image_data: BaseImages holding the stacked images and their calibration.
    """

    _required_keys = ["image_samples"]

    def __init__(
        self,
        color_type: ImageChannel = ImageChannel.RGB,
        normalize_to_unit: bool = True,
        shuffle_order: bool = False,
    ) -> None:
        """Initialize the LoadImagesFromFile transform.

        Args:
            color_type: Output color format, only rgb is supported now.
            normalize_to_unit: Whether to divide pixel values by ``255``.
            shuffle_order: Whether to load the cameras of every sample in a random order, a
                train-time regularization for models that do not depend on the camera order.
                Every per-camera field, and the image samples, follow the drawn order.
        """
        super().__init__(probability=None)
        self.color_type = color_type
        self.normalize_to_unit = normalize_to_unit
        self.shuffle_order = shuffle_order

    def load_image(self, image_path: str) -> torch.Tensor:
        """Read one image from disk as a float32 (num_channels, height, width) tensor.

        Args:
            image_path: Path of the image file.

        Returns:
            torch.Tensor: The image in channel first layout, in unit range when
              ``normalize_to_unit`` is set and in ``[0, 255]`` otherwise.
        """
        decoded_image = decode_image(image_path, mode=self.color_type.value).to(torch.float32)
        if self.normalize_to_unit:
            decoded_image = decoded_image / 255.0
        return decoded_image

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Load the images of the sample and their calibration.

        Args:
            model_gt_sample: ModelGTSample instance containing `image_samples`.

        Returns:
            Updated ModelGTSample instance with loaded `camera_image_data`.

        Raises:
            ValueError: If the sample holds no image sample.
        """
        # This is checked in the _validate_required_keys()
        image_samples = model_gt_sample.image_samples  # type: ignore[reportOptionalIterable]
        if not image_samples:
            raise ValueError("No image samples found in the ModelGTSample.")
        if self.shuffle_order:
            image_samples = [image_samples[i] for i in np.random.permutation(len(image_samples))]

        camera_intrinsics = torch.stack([sample.camera_intrinsic for sample in image_samples])
        camera_image_data = BaseImages(
            images=torch.stack([self.load_image(sample.image_path) for sample in image_samples]),
            depth_maps=None,  # No depth is loaded from an image file
            timestamps=torch.tensor(
                [sample.timestamp for sample in image_samples], dtype=torch.float64
            ),
            camera_intrinsics=camera_intrinsics,
            camera_names=[sample.camera_name for sample in image_samples],
            lidar2images=torch.stack([sample.lidar2image for sample in image_samples]),
            lidar2cams=torch.stack([sample.lidar2cam for sample in image_samples]),
            distortion_models=[sample.distortion_model for sample in image_samples],
            distortion_coefficients=[sample.distortion_coefficients for sample in image_samples],
            # No image-space transform ran yet, so the augmented intrinsics are the raw
            # ones and the composed augmentation affine is the identity.
            augmented_camera_intrinsics=camera_intrinsics.clone(),
            image_augmentation_matrices=BaseImages.identity_image_augmentation_matrices(
                camera_intrinsics
            ),
            noises=None,  # Set once the misalignment augmentation has run
        )
        return model_gt_sample._replace(
            image_samples=image_samples, camera_image_data=camera_image_data
        )
