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
import math

import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.resize import ImageSpaceTransform
from autoware_ml.types.geometry import ImageChannel


class LoadImagesFromFile(BaseTransform):
    """Load the images of every camera of the sample together with their calibration.

    The cameras are stacked along the leading dimension in the order of the image samples. The
    pixel values are served in unit range by default, which is the range the models and the
    camera-lidar fusion read.

    With ``cover_size`` every image larger than that size is downscaled, aspect kept, until it
    just covers it, while it is still the uint8 the decoder produced, and its intrinsics follow
    as after any image-space transform. The float32 conversion, the transforms that follow and
    the collate then work on a few megabytes per camera rather than on the full resolution:
    a 2880x1860 image is 64 MB as float32, and a pipeline allocating a few of those per camera
    stalls the dataloader workers in page faults long before the GPU is busy.

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
        cover_size: Sequence[int] | None = None,
    ) -> None:
        """Initialize the LoadImagesFromFile transform.

        Args:
            color_type: Output color format, only rgb is supported now.
            normalize_to_unit: Whether to divide pixel values by ``255``.
            cover_size: ``[height, width]`` the loaded images are downscaled to just cover,
                keeping their aspect ratio, or None to serve them at their stored resolution.
                An image already covered by it is served as stored, never upscaled. Set it to
                the target size of the resize that follows, which then resizes by a factor
                close to one.
        """
        super().__init__(probability=None)
        self.color_type = color_type
        self.normalize_to_unit = normalize_to_unit
        self.cover_size = tuple(cover_size) if cover_size is not None else None

    def covering_size(self, height: int, width: int) -> tuple[int, int]:
        """Size an image is downscaled to so it just covers ``cover_size``.

        Args:
            height: Height of the stored image.
            width: Width of the stored image.

        Returns:
            Downscaled ``(height, width)``, the stored size when no downscale is needed. Every
            side is rounded up, so the image still covers ``cover_size`` after rounding.
        """
        if self.cover_size is None:
            return height, width
        cover_height, cover_width = self.cover_size
        scale = max(cover_height / height, cover_width / width)
        if scale >= 1.0:
            return height, width
        return math.ceil(height * scale), math.ceil(width * scale)

    def load_image(self, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one image from disk as a float32 (num_channels, height, width) tensor.

        Args:
            image_path: Path of the image file.

        Returns:
            The image in channel first layout, in unit range when ``normalize_to_unit`` is set
            and in ``[0, 255]`` otherwise, and the 3x3 affine mapping a pixel of the stored
            image onto the returned one: the identity unless ``cover_size`` downscaled it.
        """
        # The channel orders are named after torchvision's read modes.
        read_mode = ImageReadMode[self.color_type.name]
        decoded_image = decode_image(image_path, mode=read_mode)
        height, width = decoded_image.shape[-2:]
        resized_height, resized_width = self.covering_size(height, width)
        if (resized_height, resized_width) != (height, width):
            # Resized while still uint8: a quarter of the bytes of the float32 image.
            decoded_image = v2.functional.resize(
                decoded_image, [resized_height, resized_width], antialias=True
            )
        image_transform = ImageSpaceTransform.image_scale_and_translation_matrix(
            scale_x=resized_width / width, scale_y=resized_height / height
        )
        image = decoded_image.to(torch.float32)
        if self.normalize_to_unit:
            image /= 255.0
        return image, image_transform

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

        loaded = [self.load_image(sample.image_path) for sample in image_samples]
        images = torch.stack([image for image, _ in loaded])
        image_transforms = torch.stack([image_transform for _, image_transform in loaded])

        camera_intrinsics = torch.stack([sample.camera_intrinsic for sample in image_samples])
        camera_image_data = BaseImages(
            images=images,
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
        if not torch.equal(image_transforms, torch.eye(3).expand_as(image_transforms)):
            # The downscale of ``cover_size`` is the first image-space transform of the
            # pipeline: the augmented intrinsics, the augmentation matrices and the
            # lidar2images follow it like they follow a resize.
            camera_image_data = ImageSpaceTransform.apply_image_space_transform(
                camera_image_data, images=images, image_transforms=image_transforms
            )
        return model_gt_sample._replace(camera_image_data=camera_image_data)
