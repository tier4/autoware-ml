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

"""Unit tests for image transforms."""

import numpy as np
import pytest

from autoware_ml.transforms.image.image import PhotometricDistortion


class TestPhotometricDistortion:
    @pytest.fixture
    def input_dict(self):
        # Create a random BGR image (H, W, 3) uint8
        # PhotometricDistortion is applied BEFORE LidarCameraFusion, so it expects 3-channel img
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return {"img": img}

    def test_distortion_application(self, input_dict):
        # Force probability to 1.0 to ensure transform is applied
        transform = PhotometricDistortion(
            p=1.0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
        )

        # Copy input for comparison
        img_in = input_dict["img"].copy()

        output_dict = transform(input_dict)
        img_out = output_dict["img"]

        # Check shape is preserved
        assert img_out.shape == img_in.shape

        # Check dtype is preserved
        assert img_out.dtype == np.uint8

        # Check image is modified (very unlikely to be identical with all distortions)
        assert not np.array_equal(img_out, img_in)

    def test_no_op(self, input_dict):
        # Probability 0.0 - should not modify image
        transform = PhotometricDistortion(p=0.0)
        img_in = input_dict["img"].copy()

        output_dict = transform(input_dict)
        assert np.array_equal(output_dict["img"], img_in)

    def test_preserves_other_keys(self, input_dict):
        """Test that other keys in input_dict are preserved."""
        transform = PhotometricDistortion(p=1.0, brightness=0.1)
        input_dict["other_key"] = "preserved_value"

        output_dict = transform(input_dict)

        assert "other_key" in output_dict
        assert output_dict["other_key"] == "preserved_value"

    def test_missing_img_key(self):
        """Test that missing 'img' key raises KeyError."""
        transform = PhotometricDistortion(p=1.0)

        with pytest.raises(KeyError, match="Missing required key 'img'"):
            transform({})
