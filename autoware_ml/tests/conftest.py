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

"""Global fixtures for Autoware-ML test suite."""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from autoware_ml.datamodule.t4dataset.calibration_status import (
    CalibrationData,
    CalibrationStatus,
)


@pytest.fixture
def sample_camera_matrix() -> np.ndarray:
    """Create a realistic camera intrinsic matrix (3x3)."""
    fx, fy = 1000.0, 1000.0
    cx, cy = 320.0, 240.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


@pytest.fixture
def sample_distortion_coefficients() -> np.ndarray:
    """Create sample distortion coefficients (5,)."""
    return np.array([0.1, -0.2, 0.001, 0.001, 0.05], dtype=np.float32)


@pytest.fixture
def sample_lidar_to_camera() -> np.ndarray:
    """Create a realistic LiDAR to camera transformation matrix (4x4)."""
    return np.array(
        [[0, -1, 0, 0], [0, 0, -1, -0.1], [1, 0, 0, -0.3], [0, 0, 0, 1]],
        dtype=np.float32,
    )


@pytest.fixture
def sample_calibration_data(
    sample_camera_matrix: np.ndarray,
    sample_distortion_coefficients: np.ndarray,
    sample_lidar_to_camera: np.ndarray,
) -> CalibrationData:
    """Create valid CalibrationData for testing."""
    return CalibrationData(
        camera_matrix=sample_camera_matrix,
        distortion_coefficients=sample_distortion_coefficients,
        lidar_to_camera_transformation=sample_lidar_to_camera,
    )


@pytest.fixture
def sample_calibration_data_no_distortion(
    sample_camera_matrix: np.ndarray,
    sample_lidar_to_camera: np.ndarray,
) -> CalibrationData:
    """Create CalibrationData with zero distortion for testing."""
    return CalibrationData(
        camera_matrix=sample_camera_matrix,
        distortion_coefficients=np.zeros(5, dtype=np.float32),
        lidar_to_camera_transformation=sample_lidar_to_camera,
    )


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create dummy BGR image (H, W, 3)."""
    h, w = 480, 640
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_small() -> np.ndarray:
    """Create smaller dummy BGR image for faster tests (H, W, 3)."""
    h, w = 64, 64
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.fixture
def sample_points() -> np.ndarray:
    """Create dummy point cloud (N, 5) with [x, y, z, intensity, timestamp]."""
    n_points = 1000
    xyz = np.random.rand(n_points, 3).astype(np.float32) * 50.0 - 25.0
    xyz[:, 0] += 10.0  # Offset x to be in front of camera
    intensities = np.random.rand(n_points, 1).astype(np.float32) * 255.0
    timestamps = np.zeros((n_points, 1), dtype=np.float32)
    return np.concatenate([xyz, intensities, timestamps], axis=1)


@pytest.fixture
def sample_input_dict(
    sample_image: np.ndarray,
    sample_points: np.ndarray,
    sample_calibration_data: CalibrationData,
) -> Dict[str, Any]:
    """Complete input_dict for transform testing."""
    return {
        "img": sample_image,
        "points": sample_points,
        "calibration_data": sample_calibration_data,
        "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
        "metadata": {"sample_id": "test_sample_001"},
    }


@pytest.fixture
def sample_batch_dict(
    sample_image: np.ndarray,
    sample_points: np.ndarray,
    sample_calibration_data: CalibrationData,
) -> Dict[str, List[Any]]:
    """Complete batch_dict for preprocessing testing (batch_size=2)."""
    batch_size = 2
    return {
        "img": [sample_image.copy() for _ in range(batch_size)],
        "points": [sample_points.copy() for _ in range(batch_size)],
        "calibration_data": [
            CalibrationData(
                camera_matrix=sample_calibration_data.camera_matrix.copy(),
                distortion_coefficients=sample_calibration_data.distortion_coefficients.copy(),
                lidar_to_camera_transformation=sample_calibration_data.lidar_to_camera_transformation.copy(),
            )
            for _ in range(batch_size)
        ],
        "gt_calibration_status": [CalibrationStatus.CALIBRATED.value for _ in range(batch_size)],
        "metadata": [{"sample_id": f"test_sample_{i:03d}"} for i in range(batch_size)],
    }


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()
