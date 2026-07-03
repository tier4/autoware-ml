"""Camera image resizing and cropping transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import as_hwc_image_list, restore_image_container
from autoware_ml.utils.calibration import CalibrationData


class CropAndScale(BaseTransform):
    """Crop and scale augmentation for images."""

    _required_keys = ["img", "calibration_data"]

    def __init__(self, *, p: float = 0.5, crop_ratio: float = 0.8) -> None:
        """Initialize the CropAndScale transform.

        Args:
            p: Probability of applying the transform.
            crop_ratio: Minimum fraction of the image kept when cropping.
        """
        self.p = p
        self.crop_ratio = crop_ratio

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply random crop and scale to the image."""
        image: npt.NDArray = input_dict["img"]
        calibration_data: CalibrationData = input_dict["calibration_data"]

        height, width = image.shape[:2]
        max_center_noise = (1.0 - self.crop_ratio) / 2.0
        crop_center_noise_h = self._signed_random(0, max_center_noise)
        crop_center_noise_w = self._signed_random(0, max_center_noise)
        crop_center = np.array(
            [height * (1 + crop_center_noise_h) / 2, width * (1 + crop_center_noise_w) / 2]
        )

        max_noise = max(abs(crop_center_noise_h), abs(crop_center_noise_w))
        scale_noise = np.random.uniform(self.crop_ratio, 1.0 - max_noise)
        scaled_h, scaled_w = height * scale_noise, width * scale_noise

        start_h = int(crop_center[0] - scaled_h / 2)
        end_h = int(crop_center[0] + scaled_h / 2)
        start_w = int(crop_center[1] - scaled_w / 2)
        end_w = int(crop_center[1] + scaled_w / 2)

        start_h, end_h = max(0, start_h), min(height, end_h)
        start_w, end_w = max(0, start_w), min(width, end_w)

        cropped_image = image[start_h:end_h, start_w:end_w]
        resized_image = cv2.resize(cropped_image, (width, height))

        self._update_camera_matrix(calibration_data, start_w, start_h, end_w, end_h, width)

        input_dict["img"] = resized_image
        input_dict["calibration_data"] = calibration_data
        return input_dict

    def _update_camera_matrix(
        self,
        calibration_data: CalibrationData,
        start_w: int,
        start_h: int,
        end_w: int,
        end_h: int,
        width: int,
    ) -> None:
        scale_factor = width / (end_w - start_w)
        calibration_data.new_camera_matrix[0, 0] *= scale_factor
        calibration_data.new_camera_matrix[1, 1] *= scale_factor
        calibration_data.new_camera_matrix[0, 2] = (
            calibration_data.new_camera_matrix[0, 2] - start_w
        ) * scale_factor
        calibration_data.new_camera_matrix[1, 2] = (
            calibration_data.new_camera_matrix[1, 2] - start_h
        ) * scale_factor

    def _signed_random(self, min_value: float, max_value: float) -> float:
        sign = 1 if np.random.random() < 0.5 else -1
        return sign * np.random.uniform(min_value, max_value)


class ResizeMultiviewImages(BaseTransform):
    """Resize multiview images and scale camera intrinsics consistently."""

    _required_keys = ["img", "camera_intrinsics"]

    def __init__(self, *, target_size: list[int]) -> None:
        """Initialize the ResizeMultiviewImages transform.

        Args:
            target_size: Output image size ``[height, width]``.
        """
        self.target_size = tuple(target_size)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Resize multiview images and scale intrinsics accordingly."""
        images, format_info = as_hwc_image_list(input_dict["img"])
        intrinsics = input_dict["camera_intrinsics"].copy()
        target_height, target_width = self.target_size
        intrinsics_was_single = intrinsics.ndim == 2
        if intrinsics_was_single:
            intrinsics = intrinsics[None, ...]

        resized_images = []
        for camera_index, image in enumerate(images):
            source_height, source_width = image.shape[:2]
            scale_x = target_width / source_width
            scale_y = target_height / source_height

            resized = cv2.resize(image, (target_width, target_height))
            resized_images.append(resized)

            intrinsics[camera_index, 0, 0] *= scale_x
            intrinsics[camera_index, 1, 1] *= scale_y
            intrinsics[camera_index, 0, 2] *= scale_x
            intrinsics[camera_index, 1, 2] *= scale_y

        input_dict["img"] = restore_image_container(input_dict["img"], resized_images, format_info)
        input_dict["camera_intrinsics"] = intrinsics[0] if intrinsics_was_single else intrinsics
        if "lidar2img" in input_dict:
            lidar2cam = input_dict["lidar2cam"]
            input_dict["lidar2img"] = input_dict["camera_intrinsics"] @ lidar2cam
        return input_dict


class PadMultiViewImage(BaseTransform):
    """Pad multiview images to a fixed size or size divisor."""

    _required_keys = ["img"]

    def __init__(
        self,
        *,
        size: Sequence[int] | None = None,
        size_divisor: int | None = None,
        pad_val: float = 0.0,
    ) -> None:
        """Initialize the PadMultiViewImage transform.

        Args:
            size: Optional fixed output size ``[height, width]``.
            size_divisor: Optional divisor used to round image dimensions upward.
            pad_val: Constant value used to fill padded pixels.
        """
        if size is None and size_divisor is None:
            raise ValueError("Either size or size_divisor must be provided.")
        self.size = tuple(size) if size is not None else None
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Pad one or more images."""
        images, format_info = as_hwc_image_list(input_dict["img"])
        padded = []
        for image in images:
            height, width = image.shape[:2]
            if self.size is not None:
                target_height, target_width = self.size
            else:
                target_height = int(np.ceil(height / self.size_divisor) * self.size_divisor)
                target_width = int(np.ceil(width / self.size_divisor) * self.size_divisor)
            canvas = np.full(
                (target_height, target_width, image.shape[2]), self.pad_val, dtype=image.dtype
            )
            canvas[:height, :width] = image
            padded.append(canvas)
        input_dict["img"] = restore_image_container(input_dict["img"], padded, format_info)
        input_dict["pad_shape"] = padded[0].shape[:2]
        return input_dict


class ResizeCropFlipRotImage(BaseTransform):
    """Apply resize, crop, flip, and in-plane rotation to multiview images."""

    _required_keys = ["img"]

    def __init__(
        self, *, data_aug_conf: dict[str, Any], training: bool, with_2d: bool = False
    ) -> None:
        """Initialize the ResizeCropFlipRotImage transform.

        Args:
            data_aug_conf: Augmentation config with ``final_dim``, ``resize_lim``,
                ``bot_pct_lim``, optional ``rand_flip``, and optional ``rot_lim``.
            training: Whether to sample stochastic augmentation parameters.
            with_2d: Whether 2D annotation augmentation is enabled.
        """
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.with_2d = with_2d

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Augment multiview images and update intrinsics."""
        images, format_info = as_hwc_image_list(input_dict["img"])
        augmented = []
        aug_mats = []
        intrinsics = input_dict.get("camera_intrinsics")

        for view_index, image in enumerate(images):
            transform, augmented_image = self._augment_image(image)
            augmented.append(augmented_image)
            aug_mats.append(transform)
            if intrinsics is not None:
                input_dict["camera_intrinsics"][view_index] = (
                    transform @ input_dict["camera_intrinsics"][view_index]
                )

        input_dict["img"] = restore_image_container(input_dict["img"], augmented, format_info)
        input_dict["img_aug_matrix"] = np.stack(aug_mats, axis=0).astype(np.float32)
        if intrinsics is not None and "lidar2cam" in input_dict:
            input_dict["lidar2img"] = input_dict["camera_intrinsics"] @ input_dict["lidar2cam"]
        return input_dict

    def _augment_image(self, image: npt.NDArray) -> tuple[npt.NDArray[np.float32], npt.NDArray]:
        source_height, source_width = image.shape[:2]
        final_height, final_width = self.data_aug_conf["final_dim"]
        resize_lim = self.data_aug_conf["resize_lim"]
        bot_pct_lim = self.data_aug_conf["bot_pct_lim"]

        if self.training:
            if isinstance(resize_lim, (int, float)):
                base_resize = min(final_height / source_height, final_width / source_width)
                resize = np.random.uniform(base_resize - resize_lim, base_resize + resize_lim)
            else:
                resize = np.random.uniform(*resize_lim)
            crop_bottom = np.random.uniform(*bot_pct_lim)
            crop_height = int((1 - crop_bottom) * final_height)
            crop_width = final_width
            flip = bool(self.data_aug_conf.get("rand_flip", False) and np.random.randint(2))
            rotate = float(np.random.uniform(*self.data_aug_conf.get("rot_lim", (0.0, 0.0))))
        else:
            resize = (
                min(final_height / source_height, final_width / source_width)
                if isinstance(resize_lim, (int, float))
                else float(np.mean(resize_lim))
            )
            crop_bottom = float(np.mean(bot_pct_lim))
            crop_height = int((1 - crop_bottom) * final_height)
            crop_width = final_width
            flip = False
            rotate = 0.0

        resized_width = int(source_width * resize)
        resized_height = int(source_height * resize)
        resized = cv2.resize(image, (resized_width, resized_height))

        crop_y = max(0, resized_height - crop_height)
        crop_x = max(0, (resized_width - crop_width) // 2)
        cropped = resized[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        cropped = cv2.resize(cropped, (final_width, final_height))

        transform = np.eye(4, dtype=np.float32)
        transform[0, 0] = resize * final_width / crop_width
        transform[1, 1] = resize * final_height / crop_height
        transform[0, 2] = -crop_x * final_width / crop_width
        transform[1, 2] = -crop_y * final_height / crop_height

        if flip:
            cropped = np.ascontiguousarray(np.fliplr(cropped))
            flip_mat = np.array(
                [
                    [-1.0, 0.0, final_width - 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            transform = flip_mat @ transform

        if abs(rotate) > 1e-6:
            center = (final_width / 2.0, final_height / 2.0)
            affine = cv2.getRotationMatrix2D(center, rotate, 1.0).astype(np.float32)
            cropped = cv2.warpAffine(cropped, affine, (final_width, final_height))
            rot_mat = np.eye(4, dtype=np.float32)
            rot_mat[:2, :3] = affine
            transform = rot_mat @ transform

        return transform, cropped.astype(image.dtype)
