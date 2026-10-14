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

"""Camera image resizing, cropping, and padding transforms.

Every transform of this module only rewrites pixels: the cameras are not moved and the
scene is left untouched. The pixel remapping each one applies is a 2D affine, so it is
composed into `augmented_camera_intrinsics` while `camera_intrinsics` keeps holding the
raw calibration, and `lidar2images` is recomputed from the augmented intrinsics so a 3D
point keeps projecting onto the pixel it lands on in the transformed image.

The image-space transforms are expressed in the frame of the image they are given, hence
they have to be composed in the same order as the pipeline applies them. Their composition
is kept in `image_augmentation_matrices` as 4x4 homogeneous matrices, so the pixel
remapping the augmentation applied stays available to the models that have to undo it.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


class ImageSpaceTransform(BaseTransform):
    """Base class of the transforms that only remap the pixels of the camera images.

    It holds the image-space affine builders and the composition of an image-space
    affine into the camera image data, so every subclass only has to build the affine
    matching the pixel remapping it applies.
    """

    @staticmethod
    def apply_image_space_transform(
        camera_image_data: BaseImages,
        images: Float32[Tensor, "num_cameras num_channels height width"],
        image_transforms: Float32[Tensor, "num_cameras 3 3"],
    ) -> BaseImages:
        """Rebuild camera image data whose pixels were remapped by a 2D affine.

        Args:
            camera_image_data: Camera image data holding the images before the transform.
            images: Transformed images replacing the ones held by `camera_image_data`.
            image_transforms: Per-camera 3x3 affine mapping a pixel of the input image onto
                its location in `images`.

        Returns:
            Camera image data holding the transformed images, the augmented intrinsics and
            the image augmentation matrices composed with `image_transforms`, and the
            matching `lidar2images`.
        """
        image_transforms = image_transforms.to(camera_image_data.augmented_camera_intrinsics)
        augmented_camera_intrinsics = (
            image_transforms @ camera_image_data.augmented_camera_intrinsics
        )
        # The affines are expressed in the frame of the image they are given, so the new one
        # is composed on the left of the ones the earlier transforms already applied, after
        # being lifted into the 4x4 layout the stored matrices use.
        image_augmentation_matrices = (
            BaseImages.homogeneous_image_augmentation_matrices(image_transforms)
            @ camera_image_data.image_augmentation_matrices
        )

        # (num_cameras, 4, 4), the augmented intrinsics padded to homogeneous coordinates.
        homogeneous_intrinsics = torch.eye(
            4, dtype=augmented_camera_intrinsics.dtype, device=augmented_camera_intrinsics.device
        ).repeat(augmented_camera_intrinsics.shape[0], 1, 1)
        homogeneous_intrinsics[:, :3, :3] = augmented_camera_intrinsics

        # model_copy does not validate what it is given, so the copy is validated explicitly.
        return BaseImages.model_validate(
            camera_image_data.model_copy(
                update={
                    "images": images,
                    # augmented_intrinsic @ lidar2cam -> lidar2img
                    "lidar2images": homogeneous_intrinsics @ camera_image_data.lidar2cams,
                    "augmented_camera_intrinsics": augmented_camera_intrinsics,
                    "image_augmentation_matrices": image_augmentation_matrices,
                }
            )
        )

    @staticmethod
    def image_scale_and_translation_matrix(
        scale_x: float, scale_y: float, translation_x: float = 0.0, translation_y: float = 0.0
    ) -> Float32[Tensor, "3 3"]:
        """Build the image-space affine scaling about the image origin, then translating.

        Args:
            scale_x: Scale factor along the image width.
            scale_y: Scale factor along the image height.
            translation_x: Pixel shift applied along the image width after scaling.
            translation_y: Pixel shift applied along the image height after scaling.

        Returns:
            3x3 affine mapping a pixel of the input image onto the transformed image.
        """
        return torch.tensor(
            [[scale_x, 0.0, translation_x], [0.0, scale_y, translation_y], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

    @staticmethod
    def image_horizontal_flip_matrix(width: int) -> Float32[Tensor, "3 3"]:
        """Build the image-space affine of a horizontal flip.

        Args:
            width: Width of the flipped image, whose pixel ``x`` maps onto ``width - 1 - x``.

        Returns:
            3x3 affine mapping a pixel of the input image onto the flipped image.
        """
        return torch.tensor(
            [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )

    @staticmethod
    def image_rotation_matrix(
        angle_degrees: float, height: int, width: int
    ) -> Float32[Tensor, "3 3"]:
        """Build the image-space affine of an in-plane rotation about the image center.

        Args:
            angle_degrees: Counter-clockwise rotation angle in degrees, following the
                convention of `torchvision.transforms.v2.functional.rotate`.
            height: Height of the rotated image.
            width: Width of the rotated image.

        Returns:
            3x3 affine mapping a pixel of the input image onto the rotated image.
        """
        angle_radians = np.deg2rad(angle_degrees)
        cosine, sine = float(np.cos(angle_radians)), float(np.sin(angle_radians))
        # The rotation is applied about the center of the pixel grid, which is where
        # torchvision centers it as well.
        center_x, center_y = (width - 1.0) / 2.0, (height - 1.0) / 2.0
        return torch.tensor(
            [
                [cosine, sine, center_x - cosine * center_x - sine * center_y],
                [-sine, cosine, center_y + sine * center_x - cosine * center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )


class ImageAugmentationParameters(NamedTuple):
    """Resize, crop, flip, and rotation parameters sampled for a single camera image."""

    # Size in whole pixels the source image is resized to before it is cropped.
    resized_height: int
    resized_width: int
    # Top-left corner, in pixels of the resized image, of the crop box sized as the target.
    # The box may extend past the resized image, the pixels falling outside are zero-filled.
    crop_top: int
    crop_left: int
    # Whether the crop is flipped along the image width.
    horizontal_flip: bool
    # Counter-clockwise rotation angle in degrees applied about the image center.
    rotation: float


class CropAndScale(ImageSpaceTransform):
    """Randomly crop every camera image and scale the crop back to the source size.

    The crop box is sampled independently per camera, so the field of view kept differs
    from camera to camera. Only the pixels and the augmented intrinsics change, the images
    keep the size they had before the transform.
    """

    _required_keys = ["camera_image_data"]

    def __init__(self, probability: float = 0.5, crop_ratio: float = 0.8) -> None:
        """Initialize the CropAndScale transform.

        Args:
            probability: Probability of applying the transform, None to always run.
            crop_ratio: Minimum fraction of the image kept when cropping.
        """
        super().__init__(probability=probability)
        self.crop_ratio = crop_ratio

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        """Crop and scale every camera image and update its augmented intrinsics.

        Args:
            multi_task_gt_sample: ModelGTSample instance containing `camera_image_data`.

        Returns:
            Updated ModelGTSample instance with a cropped and scaled `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        images = camera_image_data.images
        height, width = images.shape[-2:]

        cropped_images = []
        image_transforms = []
        for image in images:
            crop_top, crop_left, crop_height, crop_width = self.sample_crop_box(height, width)
            cropped = v2.functional.crop(image, crop_top, crop_left, crop_height, crop_width)
            cropped_images.append(v2.functional.resize(cropped, [height, width], antialias=True))

            scale_x, scale_y = width / crop_width, height / crop_height
            image_transforms.append(
                self.image_scale_and_translation_matrix(
                    scale_x=scale_x,
                    scale_y=scale_y,
                    translation_x=-crop_left * scale_x,
                    translation_y=-crop_top * scale_y,
                )
            )

        cropped_camera_image_data = self.apply_image_space_transform(
            camera_image_data=camera_image_data,
            images=torch.stack(cropped_images, dim=0),
            image_transforms=torch.stack(image_transforms, dim=0),
        )
        return multi_task_gt_sample._replace(camera_image_data=cropped_camera_image_data)

    def sample_crop_box(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Sample a crop box keeping at least `crop_ratio` of the image.

        Args:
            height: Height of the image the crop box is sampled for.
            width: Width of the image the crop box is sampled for.

        Returns:
            Crop box as ``(crop_top, crop_left, crop_height, crop_width)``, in pixels.
        """
        # The crop center is jittered by at most half of the pixels the crop drops, so a
        # crop keeping `crop_ratio` of the image always stays inside the image.
        max_center_noise = (1.0 - self.crop_ratio) / 2.0
        center_noise_y = self._signed_random(0.0, max_center_noise)
        center_noise_x = self._signed_random(0.0, max_center_noise)
        center_y = height * (1.0 + center_noise_y) / 2.0
        center_x = width * (1.0 + center_noise_x) / 2.0

        max_noise = max(abs(center_noise_y), abs(center_noise_x))
        crop_scale = float(np.random.uniform(self.crop_ratio, 1.0 - max_noise))

        crop_top = max(0, int(center_y - height * crop_scale / 2.0))
        crop_bottom = min(height, int(center_y + height * crop_scale / 2.0))
        crop_left = max(0, int(center_x - width * crop_scale / 2.0))
        crop_right = min(width, int(center_x + width * crop_scale / 2.0))
        return crop_top, crop_left, crop_bottom - crop_top, crop_right - crop_left

    def _signed_random(self, min_value: float, max_value: float) -> float:
        """Draw a uniform value in ``[min_value, max_value]`` with a random sign.

        Args:
            min_value: Minimum absolute value of the drawn number.
            max_value: Maximum absolute value of the drawn number.

        Returns:
            The drawn value, negated with a probability of one half.
        """
        sign = 1.0 if np.random.random() < 0.5 else -1.0
        return sign * float(np.random.uniform(min_value, max_value))


class ResizeMultiviewImages(ImageSpaceTransform):
    """Resize every camera image to a fixed size and scale the intrinsics with it."""

    _required_keys = ["camera_image_data"]

    def __init__(self, target_size: Sequence[int]) -> None:
        """Initialize the ResizeMultiviewImages transform.

        Args:
            target_size: Output image size ``[height, width]``.
        """
        super().__init__(probability=None)
        self.target_size = tuple(target_size)

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        """Resize every camera image and scale its augmented intrinsics accordingly.

        Args:
            multi_task_gt_sample: ModelGTSample instance containing `camera_image_data`.

        Returns:
            Updated ModelGTSample instance with a resized `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        images = camera_image_data.images
        source_height, source_width = images.shape[-2:]
        target_height, target_width = self.target_size

        # The leading num_cameras dimension is broadcast over automatically, every camera
        # image of a sample shares the same size.
        resized_images = v2.functional.resize(images, [target_height, target_width], antialias=True)
        image_transform = self.image_scale_and_translation_matrix(
            scale_x=target_width / source_width, scale_y=target_height / source_height
        )

        resized_camera_image_data = self.apply_image_space_transform(
            camera_image_data=camera_image_data,
            images=resized_images,
            image_transforms=image_transform.repeat(images.shape[0], 1, 1),
        )
        return multi_task_gt_sample._replace(camera_image_data=resized_camera_image_data)


class PadMultiViewImage(ImageSpaceTransform):
    """Pad every camera image to a fixed size or to a multiple of a size divisor.

    The images are padded at their bottom and right edges only, so the image origin, hence
    the intrinsics, is left untouched.
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        size: Sequence[int] | None = None,
        size_divisor: int | None = None,
        pad_value: float = 0.0,
    ) -> None:
        """Initialize the PadMultiViewImage transform.

        Args:
            size: Fixed output size ``[height, width]``, exclusive with `size_divisor`.
            size_divisor: Divisor the image dimensions are rounded upward to, exclusive
                with `size`.
            pad_value: Constant value filling the padded pixels.

        Raises:
            ValueError: If none or both of `size` and `size_divisor` are given.
        """
        if (size is None) == (size_divisor is None):
            raise ValueError("Exactly one of size or size_divisor must be provided.")
        super().__init__(probability=None)
        self.size = tuple(size) if size is not None else None
        self.size_divisor = size_divisor
        self.pad_value = pad_value

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        """Pad every camera image to the configured size.

        Args:
            multi_task_gt_sample: ModelGTSample instance containing `camera_image_data`.

        Returns:
            Updated ModelGTSample instance with a padded `camera_image_data`.

        Raises:
            ValueError: If the images are larger than the size they are padded to.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        images = camera_image_data.images
        height, width = images.shape[-2:]
        target_height, target_width = self.resolve_padded_size(height, width)
        if target_height < height or target_width < width:
            raise ValueError(
                f"{self.__class__.__name__}: cannot pad images of size "
                f"({height}, {width}) to the smaller size ({target_height}, {target_width})."
            )

        padded_images = v2.functional.pad(
            images,
            padding=[0, 0, target_width - width, target_height - height],
            fill=self.pad_value,
        )

        # Padding the bottom and the right edges keeps every pixel where it was, hence the
        # intrinsics stay valid as they are.
        padded_camera_image_data = self.apply_image_space_transform(
            camera_image_data=camera_image_data,
            images=padded_images,
            image_transforms=torch.eye(3, dtype=torch.float32).repeat(images.shape[0], 1, 1),
        )
        return multi_task_gt_sample._replace(camera_image_data=padded_camera_image_data)

    def resolve_padded_size(self, height: int, width: int) -> Tuple[int, ...]:
        """Resolve the size the images are padded to.

        Args:
            height: Height of the images before padding.
            width: Width of the images before padding.

        Returns:
            Padded image size as ``(height, width)``.
        """
        if self.size is not None:
            return self.size
        assert self.size_divisor is not None
        return (
            int(np.ceil(height / self.size_divisor) * self.size_divisor),
            int(np.ceil(width / self.size_divisor) * self.size_divisor),
        )


class ResizeCropFlipRotImage(ImageSpaceTransform):
    """Resize, crop, flip, and rotate every camera image to the target size.

    The source image is resized once, then a crop box sized as the target is cut out of it,
    so the pixels are never stretched a second time and the image transform describes the
    remapping exactly. The augmentation is sampled independently per camera. During
    validation the sampling is replaced by the mean of every configured range, the crop is
    centered along the width, and neither flip nor rotation is applied, so the transform is
    deterministic.
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        target_size: Sequence[int],
        resize_range: Sequence[float] | float,
        bottom_crop_ratio_range: Sequence[float],
        training: bool,
        random_horizontal_flip: bool = False,
        rotation_range: Sequence[float] | None = None,
    ) -> None:
        """Initialize the ResizeCropFlipRotImage transform.

        Args:
            target_size: Output image size ``[height, width]``.
            resize_range: Minimum and maximum resize factors. A single number is read as a
                tolerance around the factor covering `target_size` with the source image.
            bottom_crop_ratio_range: Minimum and maximum fraction of the resized image
                height dropped at its bottom, where the ego vehicle appears. The crop box is
                lifted by that amount and zero-filled above the top of the resized image.
            training: Whether to sample stochastic augmentation parameters.
            random_horizontal_flip: Whether the images can be flipped along their width.
            rotation_range: Minimum and maximum in-plane rotation in degrees, None to
                never rotate.
        """
        super().__init__(probability=None)
        self.target_size = tuple(target_size)
        self.resize_range = resize_range
        self.bottom_crop_ratio_range = bottom_crop_ratio_range
        self.training = training
        self.random_horizontal_flip = random_horizontal_flip
        self.rotation_range = tuple(rotation_range) if rotation_range is not None else (0.0, 0.0)

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        """Augment every camera image and update its augmented intrinsics.

        Args:
            multi_task_gt_sample: ModelGTSample instance containing `camera_image_data`.

        Returns:
            Updated ModelGTSample instance with an augmented `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        images = camera_image_data.images
        source_height, source_width = images.shape[-2:]

        augmented_images = []
        image_transforms = []
        for image in images:
            parameters = self.sample_augmentation(source_height, source_width)
            augmented_image, image_transform = self.apply_augmentation(image, parameters)
            augmented_images.append(augmented_image)
            image_transforms.append(image_transform)

        augmented_camera_image_data = self.apply_image_space_transform(
            camera_image_data=camera_image_data,
            images=torch.stack(augmented_images, dim=0),
            image_transforms=torch.stack(image_transforms, dim=0),
        )
        return multi_task_gt_sample._replace(camera_image_data=augmented_camera_image_data)

    def sample_augmentation(
        self, source_height: int, source_width: int
    ) -> ImageAugmentationParameters:
        """Sample the augmentation applied to a single camera image.

        Args:
            source_height: Height of the image before the augmentation.
            source_width: Width of the image before the augmentation.

        Returns:
            Resized size, crop box, flip, and rotation applied to the image.
        """
        target_height, target_width = self.target_size

        if isinstance(self.resize_range, (int, float)):
            # A single number is a tolerance around the factor covering the target size.
            # The covering factor (max, not min) is used so the resized image spans the
            # crop box: a smaller factor would leave zero-filled bands in the crop.
            covering_resize = max(target_height / source_height, target_width / source_width)
            resize_range = (
                covering_resize - self.resize_range,
                covering_resize + self.resize_range,
            )
        else:
            resize_range = (self.resize_range[0], self.resize_range[1])

        if self.training:
            resize = float(np.random.uniform(*resize_range))
            bottom_crop_ratio = float(
                np.random.uniform(self.bottom_crop_ratio_range[0], self.bottom_crop_ratio_range[1])
            )
            horizontal_flip = self.random_horizontal_flip and bool(np.random.randint(2))
            rotation = float(np.random.uniform(self.rotation_range[0], self.rotation_range[1]))
        else:
            resize = float(np.mean(resize_range))
            bottom_crop_ratio = float(np.mean(self.bottom_crop_ratio_range))
            horizontal_flip = False
            rotation = 0.0

        # The resized size is rounded to whole pixels, the image transform is then derived
        # from it rather than from `resize` so it holds the scale actually applied.
        resized_height = int(round(source_height * resize))
        resized_width = int(round(source_width * resize))
        # The bottom edge of the crop box sits `bottom_crop_ratio` of the resized height
        # above the bottom of the resized image, dropping the ego vehicle and keeping the
        # road. Along the width the box is placed at random in training, centered otherwise.
        crop_top = int((1.0 - bottom_crop_ratio) * resized_height) - target_height
        max_crop_left = max(0, resized_width - target_width)
        crop_left = (
            int(np.random.uniform(0, max_crop_left)) if self.training else max_crop_left // 2
        )
        return ImageAugmentationParameters(
            resized_height=resized_height,
            resized_width=resized_width,
            crop_top=crop_top,
            crop_left=crop_left,
            horizontal_flip=horizontal_flip,
            rotation=rotation,
        )

    def apply_augmentation(
        self,
        image: Float32[Tensor, "num_channels height width"],
        parameters: ImageAugmentationParameters,
    ) -> Tuple[Float32[Tensor, "num_channels height width"], Float32[Tensor, "3 3"]]:
        """Apply the sampled augmentation to a single camera image.

        Args:
            image: Single camera image the augmentation is applied to.
            parameters: Augmentation sampled for this image.

        Returns:
            The augmented image, sized as `target_size`, and the 3x3 affine mapping a pixel
            of the input image onto the augmented image.
        """
        source_height, source_width = image.shape[-2:]
        target_height, target_width = self.target_size

        resized = v2.functional.resize(
            image, [parameters.resized_height, parameters.resized_width], antialias=True
        )
        # Pixels of the crop box falling outside of the resized image are filled with zeros,
        # so the crop always has the target size without a second resize.
        augmented = v2.functional.crop(
            resized, parameters.crop_top, parameters.crop_left, target_height, target_width
        )

        # Scale the source image onto the resized one, then shift the crop box onto the
        # image origin. The scale is the one of the whole-pixel resized size.
        image_transform = self.image_scale_and_translation_matrix(
            scale_x=parameters.resized_width / source_width,
            scale_y=parameters.resized_height / source_height,
            translation_x=-parameters.crop_left,
            translation_y=-parameters.crop_top,
        )

        if parameters.horizontal_flip:
            augmented = v2.functional.horizontal_flip(augmented)
            image_transform = self.image_horizontal_flip_matrix(target_width) @ image_transform

        if abs(parameters.rotation) > 1e-6:
            augmented = v2.functional.rotate(
                augmented, parameters.rotation, interpolation=v2.InterpolationMode.BILINEAR
            )
            image_transform = (
                self.image_rotation_matrix(parameters.rotation, target_height, target_width)
                @ image_transform
            )

        return augmented.to(image.dtype), image_transform
