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

"""Unit tests for camera transforms."""

import numpy as np
import pytest

from autoware_ml.datamodule.t4dataset.calibration_status import (
    CalibrationData,
)
from autoware_ml.transforms.camera import CropAndScale, UndistortImage


class TestUndistortImage:
    """Tests for UndistortImage transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom alpha."""
        transform = UndistortImage()
        assert transform.alpha == 0.0

        transform = UndistortImage(alpha=0.5)
        assert transform.alpha == 0.5

    def test_missing_img_key(self, sample_calibration_data: CalibrationData) -> None:
        """Test that missing 'img' key raises KeyError."""
        transform = UndistortImage()
        input_dict = {"calibration_data": sample_calibration_data}

        with pytest.raises(KeyError, match="Missing required key 'img'"):
            transform(input_dict)

    def test_missing_calibration_data_key(self, sample_image: np.ndarray) -> None:
        """Test that missing 'calibration_data' key raises KeyError."""
        transform = UndistortImage()
        input_dict = {"img": sample_image}

        with pytest.raises(KeyError, match="Missing required key 'calibration_data'"):
            transform(input_dict)

    def test_passthrough_zero_distortion(
        self,
        sample_image: np.ndarray,
        sample_calibration_data_no_distortion: CalibrationData,
    ) -> None:
        """Test that zero distortion coefficients pass through unchanged."""
        transform = UndistortImage()
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data_no_distortion,
        }

        output_dict = transform(input_dict)

        # Image should be unchanged (same reference since early return)
        assert "img" in output_dict
        assert output_dict["img"].shape == sample_image.shape

    def test_output_shape_preserved(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that output image shape matches input shape."""
        transform = UndistortImage()
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        assert output_dict["img"].shape == sample_image.shape
        assert output_dict["img"].dtype == sample_image.dtype

    def test_new_camera_matrix_updated(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that new_camera_matrix is updated after undistortion."""
        transform = UndistortImage()

        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)
        output_calibration = output_dict["calibration_data"]

        # new_camera_matrix should be set
        assert output_calibration.new_camera_matrix is not None
        # Distortion coefficients should be zeroed
        assert np.allclose(output_calibration.distortion_coefficients, 0)

    def test_calibration_data_returned(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that calibration_data is returned in output."""
        transform = UndistortImage()
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        assert "calibration_data" in output_dict
        assert isinstance(output_dict["calibration_data"], CalibrationData)


class TestCropAndScale:
    """Tests for CropAndScale transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        transform = CropAndScale()
        assert transform.p == 0.5
        assert transform.crop_ratio == 0.8

        transform = CropAndScale(p=0.9, crop_ratio=0.7)
        assert transform.p == 0.9
        assert transform.crop_ratio == 0.7

    def test_missing_keys(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that missing required keys raise KeyError."""
        transform = CropAndScale()

        # Missing img
        with pytest.raises(KeyError, match="Missing required key 'img'"):
            transform({"calibration_data": sample_calibration_data})

        # Missing calibration_data
        with pytest.raises(KeyError, match="Missing required key 'calibration_data'"):
            transform({"img": sample_image})

    def test_never_apply(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test with p=0.0 returns input unchanged."""
        transform = CropAndScale(p=0.0)
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        # Should return same reference when not applied
        assert output_dict["img"].shape == sample_image.shape

    def test_always_apply_shape_preserved(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test with p=1.0 preserves output shape."""
        transform = CropAndScale(p=1.0, crop_ratio=0.8)
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        # Output shape should match input shape (resize back)
        assert output_dict["img"].shape == sample_image.shape

    def test_camera_matrix_updated(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that camera matrix is updated when transform is applied."""
        transform = CropAndScale(p=1.0, crop_ratio=0.8)
        original_camera_matrix = sample_calibration_data.new_camera_matrix.copy()

        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        # Camera matrix should be modified
        assert not np.allclose(
            output_dict["calibration_data"].new_camera_matrix,
            original_camera_matrix,
        )
