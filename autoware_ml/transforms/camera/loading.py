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

from collections.abc import Sequence

import torch
from torchvision.io import decode_image

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.geometry import ImageChannel


class LoadImagesFromFile(BaseTransform):
    """Load the images of every camera of the sample together with their calibration.

    The cameras are stacked along the leading dimension in the order of ``camera_order`` when
    given, otherwise in the order of the image samples. The pixel values are served in unit
    range by default, which is the range the models and the camera-lidar fusion read.

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
        camera_order: Sequence[str] | None = None,
    ) -> None:
        """Initialize the LoadImagesFromFile transform.

        Args:
            color_type: Output color format, only rgb is supported now.
            normalize_to_unit: Whether to divide pixel values by ``255``.
            camera_order: Names of the cameras to load, in loading order. ``None`` loads every
                camera of the sample in the order of its image samples.
        """
        super().__init__(probability=None)
        self.color_type = color_type
        self.normalize_to_unit = normalize_to_unit
        self.camera_order = list(camera_order) if camera_order is not None else None

    def select_image_samples(self, image_samples: Sequence[ImageSample]) -> list[ImageSample]:
        """Pick the image samples of the configured cameras, in loading order.

        Args:
            image_samples: Image samples of the ModelGTSample.

        Returns:
            The image samples to load, in loading order.

        Raises:
            ValueError: If a camera of ``camera_order`` is missing from the sample.
        """
        if self.camera_order is None:
            return list(image_samples)
        image_samples_by_camera_name = {
            image_sample.camera_name: image_sample for image_sample in image_samples
        }
        selected = []
        for camera_name in self.camera_order:
            if camera_name not in image_samples_by_camera_name:
                raise ValueError(
                    f"Missing camera_name: {camera_name} from the sample: {list(image_samples)}"
                )
            selected.append(image_samples_by_camera_name[camera_name])
        return selected

    def load_image(self, image_path: str) -> torch.Tensor:
        """Read one image from disk as a float32 (num_channels, height, width) tensor.

        Args:
            image_path: Path of the image file.

        Returns:
            torch.Tensor: The image in channel first layout, in unit range when
              ``normalize_to_unit`` is set and in ``[0, 255]`` otherwise.
        """
        decoded_image = decode_image(image_path, mode=self.color_type.value).to(torch.float32)  # type: ignore[arg-type]
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
        image_samples = self.select_image_samples(image_samples)

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
        return model_gt_sample._replace(camera_image_data=camera_image_data)
