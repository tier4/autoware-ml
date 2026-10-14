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

"""Photometric image transforms to support ModelGTSample."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform

# Hue is measured in degrees and the other channels in the unit range by the float HSV conversion
HUE_PERIOD_DEGREES = 360.0
CONTRAST_PIVOT = 0.5


class PhotometricDistortion(BaseTransform):
    """Apply random brightness, contrast, saturation and hue to every image.

    Operates on the camera images (num_cameras, 3, H, W) in unit range, as served by the
    loading transform. Every camera of the sample draws its own distortion.

    Required keys:
        - camera_image_data: (num_cameras, 3, H, W) float32 RGB images in [0, 1].

    Generated keys:
        - camera_image_data.images: the photometrically distorted images (when applied).
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        probability: float | None = 0.5,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        """Initialize the PhotometricDistortion transform.

        Args:
            probability: Probability of applying the transform, None to always run.
            brightness: Max brightness deviation in [0, 1].
            contrast: Max contrast deviation in [0, 1].
            saturation: Max saturation deviation in [0, 1].
            hue: Max hue deviation in [0, 0.5], as a fraction of the hue circle.
        """
        super().__init__(probability=probability)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Distort the colors of every image of the sample.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with distorted images.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        distorted = torch.stack(
            [
                torch.from_numpy(
                    np.transpose(
                        self.distort(np.transpose(image.numpy(), (1, 2, 0))), (2, 0, 1)
                    ).copy()
                )
                for image in camera_image_data.images
            ]
        )
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(update={"images": distorted})
            )
        )

    def distort(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Distort one RGB image in the unit range.

        Args:
            image: Image in (height, width, 3) layout with values in [0, 1].

        Returns:
            npt.NDArray[np.float32]: The distorted image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if self.brightness > 0:
            hsv[..., 2] *= np.random.uniform(1 - self.brightness, 1 + self.brightness)
        if self.saturation > 0:
            hsv[..., 1] *= np.random.uniform(1 - self.saturation, 1 + self.saturation)
        if self.contrast > 0:
            factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            hsv[..., 2] = (hsv[..., 2] - CONTRAST_PIVOT) * factor + CONTRAST_PIVOT
        if self.hue > 0:
            hsv[..., 0] += np.random.uniform(-self.hue, self.hue) * HUE_PERIOD_DEGREES
            hsv[..., 0] = np.mod(hsv[..., 0], HUE_PERIOD_DEGREES)

        hsv[..., 1:] = np.clip(hsv[..., 1:], 0.0, 1.0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
