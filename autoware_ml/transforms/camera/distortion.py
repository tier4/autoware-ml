"""Camera distortion correction transforms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.utils.calibration import CalibrationData


class UndistortImage(BaseTransform):
    """Undistort image using camera calibration parameters."""

    _required_keys = ["img", "calibration_data"]

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Undistort image and update calibration data."""
        image: npt.NDArray = input_dict["img"]
        calibration_data: CalibrationData = input_dict["calibration_data"]

        if not np.any(calibration_data.distortion_coefficients):
            return input_dict

        height, width = image.shape[:2]
        calibration_data.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            calibration_data.camera_matrix,
            calibration_data.distortion_coefficients,
            (width, height),
            self.alpha,
            (width, height),
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
