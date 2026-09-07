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

"""Shared calibration task types and calibration metadata containers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from jaxtyping import Float32


class CalibrationStatus(Enum):
    """Enumerate calibration-status labels used by calibration experiments."""

    MISCALIBRATED = 0
    CALIBRATED = 1


@dataclass
class CalibrationData:
    """Store camera intrinsics and lidar-to-camera calibration metadata.

    Args:
        camera_matrix: Original camera intrinsic matrix.
        distortion_coefficients: Camera distortion coefficients following OpenCV
            convention ``(k1, k2, p1, p2[, k3[, ...]])``. Length varies by model
            (4, 5, 8, 12, or 14). Empty array for pre-undistorted images.
        lidar_to_camera_transformation: Homogeneous lidar-to-camera transform.
        distortion_model: Distortion model name following ROS convention
            (e.g. ``"plumb_bob"``, ``"rational_polynomial"``). Empty string for
            pre-undistorted images.
        noise: Optional homogeneous perturbation transform applied by calibration
            augmentation.
        new_camera_matrix: Updated intrinsic matrix after image-space transforms.
            When omitted, a copy of ``camera_matrix`` is used.
    """

    camera_matrix: Float32[np.ndarray, "3 3"]
    distortion_coefficients: Float32[np.ndarray, " num_coefficients"]
    lidar_to_camera_transformation: Float32[np.ndarray, "4 4"]
    distortion_model: str = ""
    noise: Float32[np.ndarray, "4 4"] | None = None
    new_camera_matrix: Float32[np.ndarray, "3 3"] | None = None

    def __post_init__(self) -> None:
        """Initialize the post-transform intrinsic matrix if it is omitted."""
        if self.new_camera_matrix is None:
            self.new_camera_matrix = self.camera_matrix.copy()
