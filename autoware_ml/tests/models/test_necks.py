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

"""Unit tests for neck models."""

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.common.necks.global_average_pooling import GlobalAveragePooling


class TestGlobalAveragePooling:
    """Tests for GlobalAveragePooling neck."""

    def test_instantiation_default(self) -> None:
        """Test instantiation with default parameters (AdaptiveAvgPool2d)."""
        neck = GlobalAveragePooling()
        assert neck is not None
        assert isinstance(neck.gap, nn.AdaptiveAvgPool2d)

    def test_instantiation_custom_kernel(self) -> None:
        """Test instantiation with custom kernel size."""
        neck = GlobalAveragePooling(kernel_size=4, stride=4)
        assert neck is not None
        assert isinstance(neck.gap, nn.AvgPool2d)

    def test_forward_4d_tensor(self) -> None:
        """Test forward pass with 4D tensor input (B, C, H, W) -> (B, C)."""
        neck = GlobalAveragePooling()

        batch_size, channels, height, width = 4, 512, 7, 7
        input_tensor = torch.randn(batch_size, channels, height, width)

        output = neck(input_tensor)

        expected_shape = (batch_size, channels)
        assert output.shape == expected_shape

    def test_forward_varying_spatial_dims(self) -> None:
        """Test forward pass with varying spatial dimensions."""
        neck = GlobalAveragePooling()

        for h, w in [(7, 7), (14, 14), (8, 10), (1, 1)]:
            input_tensor = torch.randn(2, 256, h, w)
            output = neck(input_tensor)
            assert output.shape == (2, 256)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the neck."""
        neck = GlobalAveragePooling()

        input_tensor = torch.randn(2, 512, 7, 7, requires_grad=True)
        output = neck(input_tensor)

        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self) -> None:
        """Test forward pass on CUDA device."""
        neck = GlobalAveragePooling().cuda()

        input_tensor = torch.randn(2, 512, 7, 7).cuda()
        output = neck(input_tensor)

        assert output.device.type == "cuda"
        assert output.shape == (2, 512)
