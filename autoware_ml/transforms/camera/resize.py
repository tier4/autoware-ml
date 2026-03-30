"""Camera image resizing and cropping transforms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.utils.calibration import CalibrationData


class CropAndScale(BaseTransform):
    """Crop and scale augmentation for images."""

    _required_keys = ["img", "calibration_data"]

    def __init__(self, p: float = 0.5, crop_ratio: float = 0.8) -> None:
        self.p = p
        self.crop_ratio = crop_ratio

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply random crop and scale to the image."""
        image: npt.NDArray = input_dict["img"]
        calibration_data: CalibrationData = input_dict["calibration_data"]

        height, width = image.shape[:2]
        crop_center_noise_h = self._signed_random(0, self.crop_ratio / 2)
        crop_center_noise_w = self._signed_random(0, self.crop_ratio / 2)
        crop_center = np.array(
            [height * (1 + crop_center_noise_h) / 2, width * (1 + crop_center_noise_w) / 2]
        )

        max_noise = max(abs(crop_center_noise_h), abs(crop_center_noise_w))
        scale_noise = np.random.uniform(self.crop_ratio, 1 - max_noise)
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
