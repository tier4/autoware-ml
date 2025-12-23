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

"""Unit tests for backbone models."""

import pytest
import torch

from autoware_ml.models.common.backbones.resnet import ResNet18


class TestResNet18:
    """Tests for ResNet18 backbone."""

    def test_instantiation_default_channels(self) -> None:
        """Test instantiation with default 3 input channels."""
        model = ResNet18(in_channels=3)
        assert model is not None
        assert model.conv1.in_channels == 3

    def test_instantiation_custom_channels(self) -> None:
        """Test instantiation with custom input channels (e.g., 5 for RGBDI)."""
        model = ResNet18(in_channels=5)
        assert model is not None
        assert model.conv1.in_channels == 5

    def test_no_avgpool_fc_layers(self) -> None:
        """Test that avgpool and fc layers are removed (backbone only)."""
        model = ResNet18(in_channels=3)
        assert not hasattr(model, "avgpool")
        assert not hasattr(model, "fc")

    def test_forward_shape_standard_input(self) -> None:
        """Test forward pass shape with standard ImageNet-like input."""
        model = ResNet18(in_channels=3)
        model.eval()

        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(input_tensor)

        # ResNet18 downsamples by factor of 32: 224/32 = 7
        expected_shape = (batch_size, 512, 7, 7)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_forward_shape_custom_channels(self) -> None:
        """Test forward pass with 5-channel input (RGBDI)."""
        model = ResNet18(in_channels=5)
        model.eval()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 5, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)

        # 256/32 = 8
        expected_shape = (batch_size, 512, 8, 8)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_forward_shape_non_square_input(self) -> None:
        """Test forward pass with non-square input."""
        model = ResNet18(in_channels=3)
        model.eval()

        batch_size = 2
        height, width = 480, 640
        input_tensor = torch.randn(batch_size, 3, height, width)

        with torch.no_grad():
            output = model(input_tensor)

        # 480/32 = 15, 640/32 = 20
        expected_shape = (batch_size, 512, 15, 20)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        model = ResNet18(in_channels=3)
        model.train()

        input_tensor = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model(input_tensor)

        # Compute dummy loss and backward
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None, "Gradients should flow to input"
        assert model.conv1.weight.grad is not None, "Gradients should flow to conv1"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self) -> None:
        """Test forward pass on CUDA device."""
        model = ResNet18(in_channels=5).cuda()
        model.eval()

        input_tensor = torch.randn(2, 5, 128, 128).cuda()

        with torch.no_grad():
            output = model(input_tensor)

        assert output.device.type == "cuda"
        assert output.shape == (2, 512, 4, 4)
