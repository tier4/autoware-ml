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

"""Unit tests for the block checkpoint that keeps the batch normalization statistics honest."""

from __future__ import annotations

import copy
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from autoware_ml.models.common.activation_checkpointing import (
    checkpoint_block,
    frozen_batch_norm_statistics,
)
from autoware_ml.models.common.layers.conv import ConvModule


def _build_block() -> torch.nn.Sequential:
    """Two convolution blocks, each with a batch normalization tracking its statistics."""
    torch.manual_seed(0)
    return torch.nn.Sequential(ConvModule(4, 8), ConvModule(8, 8))


def _input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 4, 8, 8)


class TestCheckpointBlock(unittest.TestCase):
    """checkpoint_block against a plain forward and a bare non-reentrant checkpoint."""

    def test_statistics_and_gradients_match_the_plain_forward(self) -> None:
        """
        Input: the same block run plain and under checkpoint_block, one step each.
        Expected: the recomputation in the backward pass leaves the running statistics and
            the batch count where the forward put them, and the gradients are unchanged.
        Check: every buffer and every parameter gradient.
        """
        plain = _build_block().train()
        checkpointed = copy.deepcopy(plain)

        plain(_input().requires_grad_(True)).sum().backward()
        checkpoint_block(checkpointed, _input().requires_grad_(True)).sum().backward()

        for (name, buffer), (_, expected) in zip(
            checkpointed.named_buffers(), plain.named_buffers()
        ):
            torch.testing.assert_close(buffer, expected, msg=name)
        for (name, parameter), (_, expected) in zip(
            checkpointed.named_parameters(), plain.named_parameters()
        ):
            torch.testing.assert_close(parameter.grad, expected.grad, msg=name)
        self.assertEqual(int(checkpointed[0].norm.num_batches_tracked), 1)

    def test_a_bare_checkpoint_updates_the_statistics_twice(self) -> None:
        """The bare non-reentrant checkpoint is what checkpoint_block guards against."""
        block = _build_block().train()

        checkpoint(block, _input().requires_grad_(True), use_reentrant=False).sum().backward()

        self.assertEqual(int(block[0].norm.num_batches_tracked), 2)

    def test_the_momentum_is_restored_after_the_context(self) -> None:
        """The context zeroes the momentum while open and hands it back on exit."""
        block = _build_block()
        momenta = [layer.momentum for layer in block.modules() if hasattr(layer, "momentum")]

        with frozen_batch_norm_statistics(block):
            self.assertTrue(
                all(
                    layer.momentum == 0.0 for layer in block.modules() if hasattr(layer, "momentum")
                )
            )

        self.assertEqual(
            [layer.momentum for layer in block.modules() if hasattr(layer, "momentum")], momenta
        )


if __name__ == "__main__":
    unittest.main()
