# Copyright 2026 TIER IV, Inc.
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
"""Unit tests for the shared torch runtime settings."""

import unittest
from unittest.mock import patch

import spconv.constants as spconv_constants
import torch

from autoware_ml.utils.runtime import configure_torch_runtime


class TestConfigureTorchRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_matmul_precision = torch.get_float32_matmul_precision()
        self.previous_spconv_tf32 = spconv_constants.SPCONV_ALLOW_TF32

    def tearDown(self) -> None:
        torch.set_float32_matmul_precision(self.previous_matmul_precision)
        spconv_constants.SPCONV_ALLOW_TF32 = self.previous_spconv_tf32

    def test_enables_tf32_for_the_sparse_convolutions_with_the_dense_ones(self) -> None:
        spconv_constants.SPCONV_ALLOW_TF32 = False
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="mock"),
        ):
            configure_torch_runtime()
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        self.assertTrue(spconv_constants.SPCONV_ALLOW_TF32)

    def test_leaves_the_settings_alone_without_cuda(self) -> None:
        spconv_constants.SPCONV_ALLOW_TF32 = False
        with patch("torch.cuda.is_available", return_value=False):
            configure_torch_runtime()
        self.assertFalse(spconv_constants.SPCONV_ALLOW_TF32)


if __name__ == "__main__":
    unittest.main()
