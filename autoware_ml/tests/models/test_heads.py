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

"""Unit tests for head models."""

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.common.heads.linear_cls_head import ClsHead, LinearClsHead


@pytest.fixture
def default_loss() -> nn.Module:
    """Default cross-entropy loss."""
    return nn.CrossEntropyLoss()


class TestClsHead:
    """Tests for ClsHead base class."""

    def test_instantiation(self, default_loss: nn.Module) -> None:
        """Test instantiation with required parameters."""
        head = ClsHead(loss=default_loss, topk=[1], num_classes=10)
        assert head is not None
        assert isinstance(head.loss_module, nn.CrossEntropyLoss)
        assert head.topk == [1]
        assert head.cal_acc is False

    def test_pre_logits_tensor(self, default_loss: nn.Module) -> None:
        """Test pre_logits with tensor input."""
        head = ClsHead(loss=default_loss, topk=[1], num_classes=10)
        input_tensor = torch.randn(4, 512)
        output = head.pre_logits(input_tensor)
        assert torch.equal(output, input_tensor)

    def test_pre_logits_tuple(self, default_loss: nn.Module) -> None:
        """Test pre_logits with tuple input (takes last element)."""
        head = ClsHead(loss=default_loss, topk=[1], num_classes=10)
        feat1 = torch.randn(4, 256)
        feat2 = torch.randn(4, 512)
        inputs = (feat1, feat2)

        output = head.pre_logits(inputs)
        assert torch.equal(output, feat2)


class TestLinearClsHead:
    """Tests for LinearClsHead."""

    def test_instantiation(self, default_loss: nn.Module) -> None:
        """Test instantiation with required parameters."""
        head = LinearClsHead(num_classes=10, in_channels=512, loss=default_loss, topk=[1])
        assert head.num_classes == 10
        assert head.in_channels == 512
        assert head.fc.in_features == 512
        assert head.fc.out_features == 10

    def test_invalid_num_classes(self, default_loss: nn.Module) -> None:
        """Test that invalid num_classes raises error."""
        with pytest.raises(ValueError):
            LinearClsHead(num_classes=0, in_channels=512, loss=default_loss, topk=[1])

        with pytest.raises(ValueError):
            LinearClsHead(num_classes=-1, in_channels=512, loss=default_loss, topk=[1])

    def test_forward_shape(self, default_loss: nn.Module) -> None:
        """Test forward pass output shape."""
        head = LinearClsHead(num_classes=10, in_channels=512, loss=default_loss, topk=[1])

        batch_size = 4
        input_tensor = torch.randn(batch_size, 512)
        output = head(input_tensor)

        assert output.shape == (batch_size, 10)

    def test_loss_returns_dict(self, default_loss: nn.Module) -> None:
        """Test that loss method returns dict with 'loss' key."""
        head = LinearClsHead(num_classes=10, in_channels=512, loss=default_loss, topk=[1])

        input_tensor = torch.randn(4, 512)
        target = torch.randint(0, 10, (4,))
        logits = head(input_tensor)

        losses = head.loss(logits, target)

        assert isinstance(losses, dict)
        assert "loss" in losses
        assert isinstance(losses["loss"], torch.Tensor)

    def test_predict_softmax(self, default_loss: nn.Module) -> None:
        """Test that predict returns softmax probabilities."""
        head = LinearClsHead(num_classes=10, in_channels=512, loss=default_loss, topk=[1])

        input_tensor = torch.randn(4, 512)
        logits = head(input_tensor)
        output = head.predict(logits)

        # Softmax outputs should sum to 1
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_gradient_flow(self, default_loss: nn.Module) -> None:
        """Test that gradients flow through the head."""
        head = LinearClsHead(num_classes=10, in_channels=512, loss=default_loss, topk=[1])

        input_tensor = torch.randn(4, 512, requires_grad=True)
        output = head(input_tensor)

        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert head.fc.weight.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, default_loss: nn.Module) -> None:
        """Test forward pass on CUDA device."""
        head = LinearClsHead(
            num_classes=10, in_channels=512, loss=default_loss.cuda(), topk=[1]
        ).cuda()

        input_tensor = torch.randn(4, 512).cuda()
        output = head(input_tensor)

        assert output.device.type == "cuda"
        assert output.shape == (4, 10)
