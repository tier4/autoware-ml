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

"""Unit tests for camera-lidar transforms."""

from typing import Any, Dict

import numpy as np
import pytest

from autoware_ml.datamodule.t4dataset.calibration_status import (
    CalibrationData,
    CalibrationStatus,
)
from autoware_ml.transforms.camera_lidar import (
    Affine,
    CalibrationMisalignment,
    LidarCameraFusion,
)


class TestCalibrationMisalignment:
    """Tests for CalibrationMisalignment transform."""

    def test_instantiation_defaults(self) -> None:
        """Test instantiation with only required parameter (p)."""
        transform = CalibrationMisalignment(p=0.5)

        # Check probability
        assert transform.p == 0.5

        # All axes should be inactive by default
        assert transform.activate_roll is False
        assert transform.activate_pitch is False
        assert transform.activate_yaw is False
        assert transform.activate_x is False
        assert transform.activate_y is False
        assert transform.activate_z is False

        # All min/max values should be 0.0 by default
        assert transform.min_roll_neg == 0.0
        assert transform.max_roll_neg == 0.0
        assert transform.min_roll_pos == 0.0
        assert transform.max_roll_pos == 0.0

    def test_instantiation_single_axis(self) -> None:
        """Test instantiation with only one axis active."""
        transform = CalibrationMisalignment(
            p=0.5,
            activate_roll=True,
            min_roll_neg=1.0,
            max_roll_neg=5.0,
            min_roll_pos=1.0,
            max_roll_pos=5.0,
        )

        assert transform.p == 0.5
        assert transform.activate_roll is True
        assert transform.activate_pitch is False  # Still default
        assert transform.min_roll_neg == 1.0
        assert transform.max_roll_neg == 5.0
        # Other axes should remain at defaults
        assert transform.min_pitch_neg == 0.0

    def test_instantiation_all_axes(self) -> None:
        """Test instantiation with all axes active."""
        transform = CalibrationMisalignment(
            p=0.8,
            activate_roll=True,
            activate_pitch=True,
            activate_yaw=True,
            activate_x=True,
            activate_y=True,
            activate_z=True,
            min_roll_neg=1.0,
            max_roll_neg=5.0,
            min_roll_pos=1.0,
            max_roll_pos=5.0,
            min_pitch_neg=0.5,
            max_pitch_neg=2.0,
            min_pitch_pos=0.5,
            max_pitch_pos=2.0,
            min_yaw_neg=0.5,
            max_yaw_neg=2.0,
            min_yaw_pos=0.5,
            max_yaw_pos=2.0,
            min_x_neg=0.1,
            max_x_neg=0.5,
            min_x_pos=0.1,
            max_x_pos=0.5,
            min_y_neg=0.1,
            max_y_neg=0.3,
            min_y_pos=0.1,
            max_y_pos=0.3,
            min_z_neg=0.05,
            max_z_neg=0.2,
            min_z_pos=0.05,
            max_z_pos=0.2,
        )

        assert transform.p == 0.8
        assert transform.activate_roll is True
        assert transform.activate_pitch is True
        assert transform.activate_yaw is True
        assert transform.activate_x is True
        assert transform.activate_y is True
        assert transform.activate_z is True

    def test_validation_negative_values(self) -> None:
        """Test that negative magnitude values raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            CalibrationMisalignment(
                p=0.5,
                activate_roll=True,
                min_roll_neg=-1.0,  # Invalid: negative magnitude
                max_roll_neg=5.0,
            )

    def test_validation_min_greater_than_max(self) -> None:
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="must be <="):
            CalibrationMisalignment(
                p=0.5,
                activate_roll=True,
                min_roll_neg=5.0,  # Invalid: min > max
                max_roll_neg=1.0,
            )

    def test_missing_calibration_data_key(self) -> None:
        """Test that missing 'calibration_data' key raises KeyError."""
        transform = CalibrationMisalignment(p=0.5)
        input_dict = {}

        with pytest.raises(KeyError, match="Missing required key 'calibration_data'"):
            transform(input_dict)

    def test_p_zero_no_augmentation(self, sample_calibration_data: CalibrationData) -> None:
        """Test that p=0.0 applies no augmentation."""
        transform = CalibrationMisalignment(p=0.0, activate_roll=True)
        original_transform = sample_calibration_data.lidar_to_camera_transformation.copy()

        input_dict = {"calibration_data": sample_calibration_data}

        output_dict = transform(input_dict)

        # Should be unchanged
        assert np.allclose(
            output_dict["calibration_data"].lidar_to_camera_transformation,
            original_transform,
        )
        assert "gt_calibration_status" in output_dict
        assert output_dict["gt_calibration_status"] == CalibrationStatus.CALIBRATED.value

    def test_p_one_always_augment(self, sample_calibration_data: CalibrationData) -> None:
        """Test that p=1.0 always applies augmentation when axis is active."""
        transform = CalibrationMisalignment(
            p=1.0,
            activate_roll=True,
            min_roll_neg=5.0,
            max_roll_neg=10.0,
            min_roll_pos=5.0,
            max_roll_pos=10.0,
        )
        original_transform = sample_calibration_data.lidar_to_camera_transformation.copy()

        input_dict = {"calibration_data": sample_calibration_data}

        output_dict = transform(input_dict)

        # Should be different
        assert not np.allclose(
            output_dict["calibration_data"].lidar_to_camera_transformation,
            original_transform,
        )
        assert "gt_calibration_status" in output_dict
        assert output_dict["gt_calibration_status"] == CalibrationStatus.MISCALIBRATED.value

    def test_p_one_no_active_axes_still_miscalibrated(
        self, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that p=1.0 with no active axes still marks as miscalibrated but transform unchanged."""
        transform = CalibrationMisalignment(p=1.0)  # All axes inactive by default
        original_transform = sample_calibration_data.lidar_to_camera_transformation.copy()

        input_dict = {"calibration_data": sample_calibration_data}

        output_dict = transform(input_dict)

        # Transform should be unchanged (identity noise)
        assert np.allclose(
            output_dict["calibration_data"].lidar_to_camera_transformation,
            original_transform,
        )
        # But still marked as miscalibrated (augmentation was "applied")
        assert output_dict["gt_calibration_status"] == CalibrationStatus.MISCALIBRATED.value

    def test_noise_stored_in_calibration_data(
        self, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that noise transform is stored when augmentation is applied."""
        transform = CalibrationMisalignment(
            p=1.0,
            activate_roll=True,
            min_roll_neg=5.0,
            max_roll_neg=10.0,
            min_roll_pos=5.0,
            max_roll_pos=10.0,
        )

        input_dict = {"calibration_data": sample_calibration_data}
        output_dict = transform(input_dict)

        # Noise should be stored
        assert output_dict["calibration_data"].noise is not None
        assert output_dict["calibration_data"].noise.shape == (4, 4)

    def test_preserves_other_keys(self, sample_calibration_data: CalibrationData) -> None:
        """Test that all keys in input_dict are preserved."""
        transform = CalibrationMisalignment(p=0.5)
        input_dict = {
            "calibration_data": sample_calibration_data,
            "other_key": "preserved_value",
        }

        output_dict = transform(input_dict)

        assert "other_key" in output_dict
        assert output_dict["other_key"] == "preserved_value"

    def test_calibration_data_returned(self, sample_calibration_data: CalibrationData) -> None:
        """Test that calibration_data is returned in output."""
        transform = CalibrationMisalignment(p=0.5)
        input_dict = {"calibration_data": sample_calibration_data}

        output_dict = transform(input_dict)

        assert "calibration_data" in output_dict
        assert isinstance(output_dict["calibration_data"], CalibrationData)

    def test_alter_calibration_shape(self, sample_calibration_data: CalibrationData) -> None:
        """Test that alter_calibration returns correct shape."""
        transform = CalibrationMisalignment(
            p=1.0,
            activate_roll=True,
            min_roll_neg=1.0,
            max_roll_neg=5.0,
            min_roll_pos=1.0,
            max_roll_pos=5.0,
        )

        original_transform = sample_calibration_data.lidar_to_camera_transformation
        noisy_transform, noise = transform.alter_calibration(original_transform)

        assert noisy_transform.shape == (4, 4)
        assert noise.shape == (4, 4)

    def test_alter_calibration_invalid_shape(self) -> None:
        """Test that alter_calibration raises error for invalid input shape."""
        transform = CalibrationMisalignment(p=1.0, activate_roll=True)

        with pytest.raises(ValueError, match="Transform must be 4x4 matrix"):
            transform.alter_calibration(np.eye(3))

    def test_bounded_gaussian_values_in_range(self) -> None:
        """Test that bounded_gaussian produces values within specified range."""
        transform = CalibrationMisalignment(p=0.5)

        min_val, max_val = 1.0, 5.0
        center = 2.0
        scale = 1.0

        # Generate many samples
        samples = [transform.bounded_gaussian(center, min_val, max_val, scale) for _ in range(100)]

        assert all(min_val <= s <= max_val for s in samples)

    def test_bounded_gaussian_invalid_range(self) -> None:
        """Test that bounded_gaussian raises error for invalid range."""
        transform = CalibrationMisalignment(p=0.5)

        with pytest.raises(ValueError, match="min_value .* must be less than max_value"):
            transform.bounded_gaussian(center=1.0, min_value=5.0, max_value=1.0, scale=1.0)

    def test_bounded_gaussian_invalid_scale(self) -> None:
        """Test that bounded_gaussian raises error for non-positive scale."""
        transform = CalibrationMisalignment(p=0.5)

        with pytest.raises(ValueError, match="scale .* must be positive"):
            transform.bounded_gaussian(center=1.0, min_value=0.0, max_value=5.0, scale=0.0)


class TestLidarCameraFusion:
    """Tests for LidarCameraFusion transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        fusion = LidarCameraFusion()
        assert fusion.max_depth == 128.0
        assert fusion.dilation_size == 1

        fusion = LidarCameraFusion(max_depth=200.0, dilation_size=2)
        assert fusion.max_depth == 200.0
        assert fusion.dilation_size == 2

    def test_missing_img_key(
        self, sample_points: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that missing 'img' key raises KeyError."""
        fusion = LidarCameraFusion()
        input_dict = {
            "points": sample_points,
            "calibration_data": sample_calibration_data,
        }

        with pytest.raises(KeyError, match="Missing required key 'img'"):
            fusion(input_dict)

    def test_missing_points_key(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that missing 'points' key raises KeyError."""
        fusion = LidarCameraFusion()
        input_dict = {
            "img": sample_image,
            "calibration_data": sample_calibration_data,
        }

        with pytest.raises(KeyError, match="Missing required key 'points'"):
            fusion(input_dict)

    def test_missing_calibration_data_key(
        self, sample_image: np.ndarray, sample_points: np.ndarray
    ) -> None:
        """Test that missing 'calibration_data' key raises KeyError."""
        fusion = LidarCameraFusion()
        input_dict = {
            "img": sample_image,
            "points": sample_points,
        }

        with pytest.raises(KeyError, match="Missing required key 'calibration_data'"):
            fusion(input_dict)

    def test_output_fused_img_key(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that 'fused_img' key is added to output."""
        fusion = LidarCameraFusion()

        output_dict = fusion(sample_input_dict)

        assert "fused_img" in output_dict
        assert isinstance(output_dict["fused_img"], np.ndarray)

    def test_output_shape(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that fused images have correct shape (H, W, 5)."""
        fusion = LidarCameraFusion()

        output_dict = fusion(sample_input_dict)

        img = sample_input_dict["img"]
        h, w = img.shape[:2]
        assert output_dict["fused_img"].shape == (
            h,
            w,
            5,
        ), f"Expected ({h}, {w}, 5), got {output_dict['fused_img'].shape}"

    def test_output_dtype(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that fused images are float32."""
        fusion = LidarCameraFusion()

        output_dict = fusion(sample_input_dict)

        assert output_dict["fused_img"].dtype == np.float32

    def test_output_normalized(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that fused images are normalized to [0, 1]."""
        fusion = LidarCameraFusion()

        output_dict = fusion(sample_input_dict)

        fused_img = output_dict["fused_img"]
        assert fused_img.min() >= 0.0, f"Min value {fused_img.min()} < 0"
        assert fused_img.max() <= 1.0, f"Max value {fused_img.max()} > 1"

    def test_preserves_other_keys(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that all keys in input_dict are preserved."""
        fusion = LidarCameraFusion()

        output_dict = fusion(sample_input_dict)

        assert "img" in output_dict
        assert "points" in output_dict
        assert "calibration_data" in output_dict
        assert "gt_calibration_status" in output_dict

    def test_handles_affine_transform(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that affine_transform is applied when present."""
        fusion = LidarCameraFusion()

        # Add affine transform
        sample_input_dict["affine_transform"] = np.eye(3, dtype=np.float64)

        # Should not raise
        output_dict = fusion(sample_input_dict)

        assert "fused_img" in output_dict


class TestAffine:
    """Tests for Affine transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        transform = Affine()
        assert transform.p == 0.5
        assert transform.max_distortion == 0.1

        transform = Affine(p=0.8, max_distortion=0.2)
        assert transform.p == 0.8
        assert transform.max_distortion == 0.2

    def test_missing_img_key(self) -> None:
        """Test that missing 'img' key raises KeyError."""
        transform = Affine()
        input_dict = {}

        with pytest.raises(KeyError, match="Missing required key 'img'"):
            transform(input_dict)

    def test_p_zero_no_augmentation(self, sample_image: np.ndarray) -> None:
        """Test that p=0.0 applies no augmentation."""
        transform = Affine(p=0.0)
        original_image = sample_image.copy()

        input_dict = {"img": original_image}

        output_dict = transform(input_dict)

        # Image should be unchanged (same reference since early return)
        assert "img" in output_dict
        assert output_dict["img"] is original_image

        # Check that affine_transform is present and is identity
        assert "affine_transform" in output_dict
        assert np.allclose(output_dict["affine_transform"], np.eye(3))

    def test_p_one_always_augment(self, sample_image: np.ndarray) -> None:
        """Test that p=1.0 always applies augmentation."""
        transform = Affine(p=1.0)
        original_image = sample_image.copy()

        input_dict = {"img": original_image}

        output_dict = transform(input_dict)

        # Should be different and affine_transform should be added
        assert "img" in output_dict
        assert "affine_transform" in output_dict
        assert output_dict["affine_transform"].shape == (3, 3)

    def test_preserves_other_keys(self, sample_image: np.ndarray) -> None:
        """Test that all keys in input_dict are preserved."""
        transform = Affine()
        input_dict = {"img": sample_image.copy(), "other_key": "preserved_value"}

        output_dict = transform(input_dict)

        assert "other_key" in output_dict
        assert output_dict["other_key"] == "preserved_value"

    def test_output_shape_preserved(self, sample_image: np.ndarray) -> None:
        """Test that output image shape matches input shape."""
        transform = Affine()
        input_dict = {"img": sample_image.copy()}

        output_dict = transform(input_dict)

        assert output_dict["img"].shape == sample_image.shape
        assert output_dict["img"].dtype == sample_image.dtype

    def test_affine_matrix_valid(self, sample_image: np.ndarray) -> None:
        """Test that affine matrix has valid structure."""
        transform = Affine(p=1.0, max_distortion=0.1)
        input_dict = {"img": sample_image.copy()}

        output_dict = transform(input_dict)

        affine = output_dict["affine_transform"]
        # Last row should be [0, 0, 1] for affine transform
        assert np.allclose(affine[2, :], [0, 0, 1])
