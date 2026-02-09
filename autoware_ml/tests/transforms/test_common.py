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

"""Unit tests for PermuteAxes transforms."""

from typing import Any, Dict

import numpy as np
import pytest
import torch

from autoware_ml.transforms.common import PermuteAxes


class TestPermuteAxes:
    """Tests for PermuteAxes transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with input_keys and axes."""
        permute = PermuteAxes(input_keys=["data"], axes=(2, 0, 1))
        assert permute.input_keys == ["data"]
        assert permute.axes == (2, 0, 1)

    def test_missing_key(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that missing required key raises KeyError."""
        permute = PermuteAxes(input_keys=["missing_key"], axes=(0, 1))
        with pytest.raises(KeyError, match="Missing required key 'missing_key'"):
            permute(sample_input_dict)

    def test_permute_numpy_hwc_to_chw(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test permuting numpy array from (H, W, C) to (C, H, W)."""
        sample_input_dict["test_array"] = np.random.rand(32, 64, 5).astype(np.float32)

        permute = PermuteAxes(input_keys=["test_array"], axes=(2, 0, 1))
        output_dict = permute(sample_input_dict)

        assert output_dict["test_array"].shape == (5, 32, 64)

    def test_permute_tensor_hwc_to_chw(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test permuting tensor from (H, W, C) to (C, H, W)."""
        sample_input_dict["test_tensor"] = torch.randn(32, 64, 5)

        permute = PermuteAxes(input_keys=["test_tensor"], axes=(2, 0, 1))
        output_dict = permute(sample_input_dict)

        assert output_dict["test_tensor"].shape == (5, 32, 64)

    def test_multiple_keys(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test permuting multiple keys."""
        sample_input_dict["img1"] = np.random.rand(32, 64, 3).astype(np.float32)
        sample_input_dict["img2"] = np.random.rand(16, 32, 5).astype(np.float32)

        permute = PermuteAxes(input_keys=["img1", "img2"], axes=(2, 0, 1))
        output_dict = permute(sample_input_dict)

        assert output_dict["img1"].shape == (3, 32, 64)
        assert output_dict["img2"].shape == (5, 16, 32)

    def test_preserves_other_keys(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test that other keys are preserved."""
        sample_input_dict["test_array"] = np.random.rand(32, 64, 5).astype(np.float32)
        sample_input_dict["other_data"] = "preserved"

        permute = PermuteAxes(input_keys=["test_array"], axes=(2, 0, 1))
        output_dict = permute(sample_input_dict)

        assert output_dict["test_array"].shape == (5, 32, 64)
        assert output_dict["other_data"] == "preserved"

    def test_identity_permutation(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test identity permutation."""
        array = np.random.rand(32, 64, 5).astype(np.float32)
        sample_input_dict["test_array"] = array.copy()

        permute = PermuteAxes(input_keys=["test_array"], axes=(0, 1, 2))
        output_dict = permute(sample_input_dict)

        assert np.array_equal(output_dict["test_array"], array)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_permute(self, sample_input_dict: Dict[str, Any]) -> None:
        """Test permuting CUDA tensor."""
        sample_input_dict["test_tensor"] = torch.randn(32, 64, 5).cuda()

        permute = PermuteAxes(input_keys=["test_tensor"], axes=(2, 0, 1))
        output_dict = permute(sample_input_dict)

        assert output_dict["test_tensor"].device.type == "cuda"
        assert output_dict["test_tensor"].shape == (5, 32, 64)
