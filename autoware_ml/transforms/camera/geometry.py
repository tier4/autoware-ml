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

"""Image space geometry transforms to support ModelGTSample."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.transforms.base import BaseTransform


def crop_with_zero_padding(
    image: npt.NDArray, crop_x: int, crop_y: int, width: int, height: int
) -> npt.NDArray:
    """Crop a window from an image, zero filling any area outside its bounds.

    The window may extend past the image edges. Pixels outside the image stay zero.

    Args:
        image: Image in (height, width, num_channels) layout.
        crop_x: Left edge of the window in the source image.
        crop_y: Top edge of the window in the source image.
        width: Width of the window.
        height: Height of the window.

    Returns:
        npt.NDArray: The cropped window.
    """
    canvas = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
    source_y0 = min(max(crop_y, 0), image.shape[0])
    source_y1 = min(max(crop_y + height, 0), image.shape[0])
    source_x0 = min(max(crop_x, 0), image.shape[1])
    source_x1 = min(max(crop_x + width, 0), image.shape[1])
    canvas[source_y0 - crop_y : source_y1 - crop_y, source_x0 - crop_x : source_x1 - crop_x] = (
        image[source_y0:source_y1, source_x0:source_x1]
    )
    return canvas


class ImageAug3D(BaseTransform):
    """Resize, crop, flip and rotate the images and follow it in their calibration.

    The pixel affine of every camera is composed into the augmentation matrices and into the
    augmented intrinsics, so the view transform projects the points onto the augmented image
    rather than onto the raw one.
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        final_dim: Sequence[int],
        resize_lim: Sequence[float],
        bot_pct_lim: Sequence[float],
        rand_flip: bool = False,
        rot_lim: Sequence[float] | None = None,
        training: bool = True,
    ) -> None:
        """Initialize the ImageAug3D transform.

        Args:
            final_dim: Final image size [height, width].
            resize_lim: Minimum and maximum resize factors.
            bot_pct_lim: Bottom crop ratios.
            rand_flip: Whether horizontal flipping is enabled.
            rot_lim: In plane rotation range in degrees, None to keep the images upright.
            training: Whether to sample stochastic augmentation parameters.
        """
        super().__init__(probability=None)
        self.final_dim = tuple(final_dim)
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = tuple(rot_lim) if rot_lim is not None else (0.0, 0.0)
        self.training = training

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Augment every image of the sample and update its calibration.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with augmented images.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        augmented_images = []
        pixel_affines = []
        for image in camera_image_data.images:
            pixel_affine, augmented_image = self.augment_image(
                np.transpose(image.numpy(), (1, 2, 0))
            )
            augmented_images.append(
                torch.from_numpy(np.transpose(augmented_image, (2, 0, 1)).copy())
            )
            pixel_affines.append(torch.from_numpy(pixel_affine))

        stacked_affines = torch.stack(pixel_affines)
        augmentation_matrices = BaseImages.homogeneous_image_augmentation_matrices(stacked_affines)
        augmented_camera_intrinsics = (
            stacked_affines @ camera_image_data.augmented_camera_intrinsics
        )
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(
                    update={
                        "images": torch.stack(augmented_images),
                        "augmented_camera_intrinsics": augmented_camera_intrinsics,
                        "image_augmentation_matrices": augmentation_matrices
                        @ camera_image_data.image_augmentation_matrices,
                        "lidar2images": compose_lidar2images(
                            augmented_camera_intrinsics, camera_image_data.lidar2cams
                        ),
                    }
                )
            )
        )

    def sample_augmentation(self) -> tuple[float, float, bool, float]:
        """Draw the resize factor, the bottom crop, the flip and the rotation of one image.

        Returns:
            tuple[float, float, bool, float]: Resize factor, bottom crop ratio, flip flag and
                rotation in degrees.
        """
        if not self.training:
            return float(np.mean(self.resize_lim)), float(np.mean(self.bot_pct_lim)), False, 0.0

        resize = float(np.random.uniform(*self.resize_lim))
        crop_bottom = float(np.random.uniform(*self.bot_pct_lim))
        flip = bool(self.rand_flip and np.random.randint(2))
        rotate = float(np.random.uniform(*self.rot_lim))
        return resize, crop_bottom, flip, rotate

    def augment_image(self, image: npt.NDArray) -> tuple[npt.NDArray[np.float32], npt.NDArray]:
        """Augment one image and return the pixel affine it was augmented with.

        Args:
            image: Image in (height, width, num_channels) layout.

        Returns:
            tuple[npt.NDArray[np.float32], npt.NDArray]: The 3x3 pixel affine and the
                augmented image.
        """
        source_height, source_width = image.shape[:2]
        final_height, final_width = self.final_dim
        resize, crop_bottom, flip, rotate = self.sample_augmentation()

        resized_width = int(round(source_width * resize))
        resized_height = int(round(source_height * resize))
        resized = cv2.resize(image, (resized_width, resized_height))

        crop_y = int((1.0 - crop_bottom) * resized_height) - final_height
        max_crop_x = max(0, resized_width - final_width)
        crop_x = int(np.random.uniform(0, max_crop_x)) if self.training else max_crop_x // 2
        cropped = crop_with_zero_padding(resized, crop_x, crop_y, final_width, final_height)

        pixel_affine = np.eye(3, dtype=np.float32)
        pixel_affine[0, 0] = resized_width / source_width
        pixel_affine[1, 1] = resized_height / source_height
        pixel_affine[0, 2] = -crop_x
        pixel_affine[1, 2] = -crop_y

        if flip:
            cropped = np.ascontiguousarray(np.fliplr(cropped))
            pixel_affine = (
                np.array(
                    [[-1.0, 0.0, final_width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                @ pixel_affine
            )

        if abs(rotate) > 1e-6:
            center = (final_width / 2.0, final_height / 2.0)
            affine = cv2.getRotationMatrix2D(center, rotate, 1.0).astype(np.float32)
            cropped = cv2.warpAffine(cropped, affine, (final_width, final_height))
            rotation = np.eye(3, dtype=np.float32)
            rotation[:2, :3] = affine
            pixel_affine = rotation @ pixel_affine

        return pixel_affine, cropped.astype(image.dtype, copy=False)


class CropAndScale(BaseTransform):
    """Crop a random window out of every image and scale it back to the full size."""

    _required_keys = ["camera_image_data"]

    def __init__(self, probability: float = 0.5, crop_ratio: float = 0.8) -> None:
        """Initialize the CropAndScale transform.

        Args:
            probability: Probability of applying the transform.
            crop_ratio: Minimum fraction of the image kept when cropping.
        """
        super().__init__(probability=probability)
        self.crop_ratio = crop_ratio

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Crop and scale every image of the sample and update its calibration.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with cropped images.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        images = []
        pixel_affines = []
        for image in camera_image_data.images:
            pixel_affine, cropped = self.crop_and_scale(np.transpose(image.numpy(), (1, 2, 0)))
            images.append(torch.from_numpy(np.transpose(cropped, (2, 0, 1)).copy()))
            pixel_affines.append(torch.from_numpy(pixel_affine))

        stacked_affines = torch.stack(pixel_affines)
        augmented_camera_intrinsics = (
            stacked_affines @ camera_image_data.augmented_camera_intrinsics
        )
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(
                    update={
                        "images": torch.stack(images),
                        # The intrinsics follow the crop, so the projected points land on
                        # the cropped pixels without a further pixel affine
                        "augmented_camera_intrinsics": augmented_camera_intrinsics,
                        "lidar2images": compose_lidar2images(
                            augmented_camera_intrinsics, camera_image_data.lidar2cams
                        ),
                    }
                )
            )
        )

    def signed_random(self, min_value: float, max_value: float) -> float:
        """Draw a value of random sign from the given magnitude range.

        Args:
            min_value: Smallest magnitude.
            max_value: Largest magnitude.

        Returns:
            float: The drawn value.
        """
        sign = 1 if np.random.random() < 0.5 else -1
        return sign * np.random.uniform(min_value, max_value)

    def crop_and_scale(
        self, image: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Crop one image and scale it back, returning the pixel affine it took.

        Args:
            image: Image in (height, width, num_channels) layout.

        Returns:
            tuple: The 3x3 pixel affine and the cropped and scaled image.
        """
        height, width = image.shape[:2]
        max_center_noise = (1.0 - self.crop_ratio) / 2.0
        crop_center_noise_h = self.signed_random(0, max_center_noise)
        crop_center_noise_w = self.signed_random(0, max_center_noise)
        crop_center = np.array(
            [height * (1 + crop_center_noise_h) / 2, width * (1 + crop_center_noise_w) / 2]
        )

        max_noise = max(abs(crop_center_noise_h), abs(crop_center_noise_w))
        scale_noise = np.random.uniform(self.crop_ratio, 1.0 - max_noise)
        scaled_h, scaled_w = height * scale_noise, width * scale_noise

        start_h = max(0, int(crop_center[0] - scaled_h / 2))
        end_h = min(height, int(crop_center[0] + scaled_h / 2))
        start_w = max(0, int(crop_center[1] - scaled_w / 2))
        end_w = min(width, int(crop_center[1] + scaled_w / 2))

        resized = cv2.resize(image[start_h:end_h, start_w:end_w], (width, height))

        scale_factor = width / (end_w - start_w)
        pixel_affine = np.eye(3, dtype=np.float32)
        pixel_affine[0, 0] = scale_factor
        pixel_affine[1, 1] = scale_factor
        pixel_affine[0, 2] = -start_w * scale_factor
        pixel_affine[1, 2] = -start_h * scale_factor
        return pixel_affine, resized
