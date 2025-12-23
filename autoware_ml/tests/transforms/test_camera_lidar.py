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
    CalibrationMisalignment,
    LidarCameraFusion,
    RandomAffine,
)


class TestCalibrationMisalignment:
    """Tests for CalibrationMisalignment transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        transform = CalibrationMisalignment()
        assert transform.p == 0.5
        assert transform.min_angle == 1.0
        assert transform.max_angle == 10.0

        transform = CalibrationMisalignment(p=0.8, min_angle=2.0, max_angle=5.0)
        assert transform.p == 0.8
        assert transform.min_angle == 2.0
        assert transform.max_angle == 5.0

    def test_missing_calibration_data_key(self) -> None:
        """Test that missing 'calibration_data' key raises assertion error."""
        transform = CalibrationMisalignment()
        input_dict = {}

        with pytest.raises(AssertionError, match="Missing required key: 'calibration_data'"):
            transform(input_dict)

    def test_p_zero_no_augmentation(self, sample_calibration_data: CalibrationData) -> None:
        """Test that p=0.0 applies no augmentation."""
        transform = CalibrationMisalignment(p=0.0)
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
        """Test that p=1.0 always applies augmentation."""
        transform = CalibrationMisalignment(p=1.0)
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

    def test_preserves_other_keys(self, sample_calibration_data: CalibrationData) -> None:
        """Test that all keys in input_dict are preserved."""
        transform = CalibrationMisalignment()
        input_dict = {
            "calibration_data": sample_calibration_data,
            "other_key": "preserved_value",
        }

        output_dict = transform(input_dict)

        assert "other_key" in output_dict
        assert output_dict["other_key"] == "preserved_value"

    def test_calibration_data_returned(self, sample_calibration_data: CalibrationData) -> None:
        """Test that calibration_data is returned in output."""
        transform = CalibrationMisalignment()
        input_dict = {"calibration_data": sample_calibration_data}

        output_dict = transform(input_dict)

        assert "calibration_data" in output_dict
        assert isinstance(output_dict["calibration_data"], CalibrationData)


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
        """Test that missing 'img' key raises assertion error."""
        fusion = LidarCameraFusion()
        input_dict = {
            "points": sample_points,
            "calibration_data": sample_calibration_data,
        }

        with pytest.raises(AssertionError, match="Missing required key: 'img'"):
            fusion(input_dict)

    def test_missing_points_key(
        self, sample_image: np.ndarray, sample_calibration_data: CalibrationData
    ) -> None:
        """Test that missing 'points' key raises assertion error."""
        fusion = LidarCameraFusion()
        input_dict = {
            "img": sample_image,
            "calibration_data": sample_calibration_data,
        }

        with pytest.raises(AssertionError, match="Missing required key: 'points'"):
            fusion(input_dict)

    def test_missing_calibration_data_key(
        self, sample_image: np.ndarray, sample_points: np.ndarray
    ) -> None:
        """Test that missing 'calibration_data' key raises assertion error."""
        fusion = LidarCameraFusion()
        input_dict = {
            "img": sample_image,
            "points": sample_points,
        }

        with pytest.raises(AssertionError, match="Missing required key: 'calibration_data'"):
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


class TestRandomAffine:
    """Tests for RandomAffine transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        transform = RandomAffine()
        assert transform.p == 0.5
        assert transform.max_distortion == 0.1

        transform = RandomAffine(p=0.8, max_distortion=0.2)
        assert transform.p == 0.8
        assert transform.max_distortion == 0.2

    def test_missing_img_key(self) -> None:
        """Test that missing 'img' key raises assertion error."""
        transform = RandomAffine()
        input_dict = {}

        with pytest.raises(AssertionError, match="Missing required key: 'img'"):
            transform(input_dict)

    def test_p_zero_no_augmentation(self, sample_image: np.ndarray) -> None:
        """Test that p=0.0 applies no augmentation."""
        transform = RandomAffine(p=0.0)
        original_image = sample_image.copy()

        input_dict = {"img": original_image}

        output_dict = transform(input_dict)

        # Image should be unchanged (same reference since early return)
        assert "img" in output_dict
        assert output_dict["img"] is original_image

    def test_p_one_always_augment(self, sample_image: np.ndarray) -> None:
        """Test that p=1.0 always applies augmentation."""
        transform = RandomAffine(p=1.0)
        original_image = sample_image.copy()

        input_dict = {"img": original_image}

        output_dict = transform(input_dict)

        # Should be different and affine_transform should be added
        assert "img" in output_dict
        assert "affine_transform" in output_dict
        assert output_dict["affine_transform"].shape == (3, 3)

    def test_preserves_other_keys(self, sample_image: np.ndarray) -> None:
        """Test that all keys in input_dict are preserved."""
        transform = RandomAffine()
        input_dict = {"img": sample_image.copy(), "other_key": "preserved_value"}

        output_dict = transform(input_dict)

        assert "other_key" in output_dict
        assert output_dict["other_key"] == "preserved_value"

    def test_output_shape_preserved(self, sample_image: np.ndarray) -> None:
        """Test that output image shape matches input shape."""
        transform = RandomAffine()
        input_dict = {"img": sample_image.copy()}

        output_dict = transform(input_dict)

        assert output_dict["img"].shape == sample_image.shape
        assert output_dict["img"].dtype == sample_image.dtype

    def test_affine_matrix_valid(self, sample_image: np.ndarray) -> None:
        """Test that affine matrix has valid structure."""
        transform = RandomAffine(p=1.0, max_distortion=0.1)
        input_dict = {"img": sample_image.copy()}

        output_dict = transform(input_dict)

        affine = output_dict["affine_transform"]
        # Last row should be [0, 0, 1] for affine transform
        assert np.allclose(affine[2, :], [0, 0, 1])
