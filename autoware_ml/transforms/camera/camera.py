# Copyright 2025 TIER IV, Inc.
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

"""Camera-specific transforms for image and calibration data."""

from typing import Any, Dict

import cv2
import numpy as np

from autoware_ml.datamodule.t4dataset.calibration_status import CalibrationData
from autoware_ml.transforms.base import BaseTransform


class UndistortImage(BaseTransform):
    """Undistort image using camera calibration parameters.

    Applies lens distortion correction using OpenCV's undistort function
    and updates the calibration data with the new camera matrix.

    Required keys:
        - img: (H, W, 3) uint8 image.
        - calibration_data: CalibrationData object with camera_matrix and
          distortion_coefficients.

    Optional keys:
        - None

    Generated keys:
        - img: Modified in-place with undistorted image (when distortion exists).
        - calibration_data.new_camera_matrix: Updated with optimal camera matrix.
        - calibration_data.distortion_coefficients: Set to zeros after undistortion.

    Args:
        alpha: Free scaling parameter between 0 and 1.
            0: Crops all invalid pixels (no black borders).
            1: Retains all pixels (may have black borders).
    """

    _required_keys = ["img", "calibration_data"]

    def __init__(self, alpha: float = 0.0) -> None:
        """Initialize UndistortImage transform.

        Args:
            alpha: Free scaling parameter between 0 and 1.
        """
        super().__init__()
        self.alpha = alpha

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Undistort image and update calibration data.

        Args:
            input_dict: Dictionary with 'img' and 'calibration_data'.

        Returns:
            Dictionary with undistorted image and updated calibration data.
        """
        image: np.ndarray = input_dict["img"]
        calibration_data: CalibrationData = input_dict["calibration_data"]

        if not np.any(calibration_data.distortion_coefficients):
            return input_dict

        h, w = image.shape[:2]
        calibration_data.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            calibration_data.camera_matrix,
            calibration_data.distortion_coefficients,
            (w, h),
            self.alpha,
            (w, h),
        )
        image = cv2.undistort(
            image,
            calibration_data.camera_matrix,
            calibration_data.distortion_coefficients,
            newCameraMatrix=calibration_data.new_camera_matrix,
        )
        calibration_data.distortion_coefficients = np.zeros_like(
            calibration_data.distortion_coefficients
        )
        input_dict["img"] = image
        input_dict["calibration_data"] = calibration_data
        return input_dict


class CropAndScale(BaseTransform):
    """Crop and scale augmentation for images.

    Crops a random region from the image and resizes back to original dimensions.
    Updates camera matrix to account for the transformation.

    Required keys:
        - img: (H, W, 3) uint8 image.
        - calibration_data: CalibrationData object with new_camera_matrix.

    Optional keys:
        - None

    Generated keys:
        - img: Modified in-place with cropped and scaled image (when applied).
        - calibration_data.new_camera_matrix: Updated to account for crop/scale.

    Args:
        p: Probability of applying augmentation (0.0 to 1.0).
        crop_ratio: Minimum crop ratio controlling crop size range (0.0 to 1.0).
    """

    _required_keys = ["img", "calibration_data"]

    def __init__(self, p: float = 0.5, crop_ratio: float = 0.8) -> None:
        """Initialize CropAndScale transform.

        Args:
            p: Probability of applying augmentation.
            crop_ratio: Minimum crop ratio controlling crop size range.
        """
        super().__init__()
        self.p = p
        self.crop_ratio = crop_ratio

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random crop and scale to the image.

        Args:
            input_dict: Dictionary with 'img' and 'calibration_data'.

        Returns:
            Dictionary with cropped/scaled image and updated calibration data.
        """
        image: np.ndarray = input_dict["img"]
        calibration_data: CalibrationData = input_dict["calibration_data"]

        h, w = image.shape[:2]

        crop_center_noise_h = self._signed_random(0, self.crop_ratio / 2)
        crop_center_noise_w = self._signed_random(0, self.crop_ratio / 2)
        crop_center = np.array(
            [h * (1 + crop_center_noise_h) / 2, w * (1 + crop_center_noise_w) / 2]
        )

        max_noise = max(abs(crop_center_noise_h), abs(crop_center_noise_w))
        scale_noise = np.random.uniform(self.crop_ratio, 1 - max_noise)
        scaled_h, scaled_w = h * scale_noise, w * scale_noise

        start_h = int(crop_center[0] - scaled_h / 2)
        end_h = int(crop_center[0] + scaled_h / 2)
        start_w = int(crop_center[1] - scaled_w / 2)
        end_w = int(crop_center[1] + scaled_w / 2)

        start_h, end_h = max(0, start_h), min(h, end_h)
        start_w, end_w = max(0, start_w), min(w, end_w)

        cropped_image = image[start_h:end_h, start_w:end_w]
        resized_image = cv2.resize(cropped_image, (w, h))

        self._update_camera_matrix(calibration_data, start_w, start_h, end_w, end_h, w)

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
        w: int,
    ) -> None:
        """Update camera matrix to account for cropping and scaling."""
        scale_factor = w / (end_w - start_w)
        calibration_data.new_camera_matrix[0, 0] *= scale_factor  # fx
        calibration_data.new_camera_matrix[1, 1] *= scale_factor  # fy
        calibration_data.new_camera_matrix[0, 2] = (
            calibration_data.new_camera_matrix[0, 2] - start_w
        ) * scale_factor  # cx
        calibration_data.new_camera_matrix[1, 2] = (
            calibration_data.new_camera_matrix[1, 2] - start_h
        ) * scale_factor  # cy

    def _signed_random(self, min_value: float, max_value: float) -> float:
        """Generate random value with random sign."""
        sign = 1 if np.random.random() < 0.5 else -1
        return sign * np.random.uniform(min_value, max_value)
