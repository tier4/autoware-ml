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

"""Unit tests for the SECOND backbone: output shapes and the activation checkpointing path."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from autoware_ml.models.common import activation_checkpointing
from autoware_ml.models.detection3d.backbones import second
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone

_IN_CHANNELS = 4
_OUT_CHANNELS = (8, 16)
_LAYER_NUMS = (1, 2)
_LAYER_STRIDES = (1, 2)
_BATCH_SIZE = 2
_BEV_SIZE = 16


def _build_backbone(activation_checkpointing: bool) -> SECONDBackbone:
    """Build a two-stage backbone from a fixed seed."""
    torch.manual_seed(0)
    return SECONDBackbone(
        in_channels=_IN_CHANNELS,
        out_channels=_OUT_CHANNELS,
        layer_nums=_LAYER_NUMS,
        layer_strides=_LAYER_STRIDES,
        activation_checkpointing=activation_checkpointing,
    )


def _build_backbone_pair() -> tuple[SECONDBackbone, SECONDBackbone]:
    """Build a plain and a checkpointed backbone sharing the same weights."""
    plain = _build_backbone(activation_checkpointing=False)
    checkpointed = _build_backbone(activation_checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    return plain, checkpointed


def _input() -> torch.Tensor:
    """Build a fixed BEV feature map."""
    torch.manual_seed(1)
    return torch.randn(_BATCH_SIZE, _IN_CHANNELS, _BEV_SIZE, _BEV_SIZE)


def _loss(outputs: list[torch.Tensor]) -> torch.Tensor:
    """Reduce the multi-scale outputs to a scalar that depends on every element."""
    return sum(output.sum() for output in outputs)


class TestSECONDBackbone(unittest.TestCase):
    """Unit tests for the SECOND backbone forward pass."""

    def test_returns_one_feature_map_per_stage(self) -> None:
        """Each stage halves the resolution by its stride and widens to its channel count."""
        backbone = _build_backbone(activation_checkpointing=False).eval()

        with torch.no_grad():
            stage1, stage2 = backbone(_input())

        self.assertEqual(stage1.shape, (_BATCH_SIZE, 8, _BEV_SIZE, _BEV_SIZE))
        self.assertEqual(stage2.shape, (_BATCH_SIZE, 16, _BEV_SIZE // 2, _BEV_SIZE // 2))


class TestActivationCheckpointing(unittest.TestCase):
    """Activation checkpointing trades memory for recomputation without changing the math."""

    def test_eval_outputs_match_without_checkpointing(self) -> None:
        """In eval mode the checkpointed outputs equal the plain outputs."""
        plain, checkpointed = _build_backbone_pair()
        plain.eval()
        checkpointed.eval()

        with torch.no_grad():
            plain_outputs = plain(_input())
            checkpointed_outputs = checkpointed(_input())

        self.assertEqual(len(plain_outputs), len(checkpointed_outputs))
        for plain_output, checkpointed_output in zip(plain_outputs, checkpointed_outputs):
            torch.testing.assert_close(checkpointed_output, plain_output)

    def test_train_outputs_match_without_checkpointing(self) -> None:
        """In train mode batch statistics drive the normalization and the outputs still match."""
        plain, checkpointed = _build_backbone_pair()
        plain.train()
        checkpointed.train()

        plain_outputs = plain(_input())
        checkpointed_outputs = checkpointed(_input())

        self.assertEqual(len(plain_outputs), len(checkpointed_outputs))
        for plain_output, checkpointed_output in zip(plain_outputs, checkpointed_outputs):
            torch.testing.assert_close(checkpointed_output, plain_output)

    def test_input_gradients_match_without_checkpointing(self) -> None:
        """The recomputed backward pass yields the same input gradient."""
        plain, checkpointed = _build_backbone_pair()
        plain.train()
        checkpointed.train()
        plain_input = _input().requires_grad_(True)
        checkpointed_input = _input().requires_grad_(True)

        _loss(plain(plain_input)).backward()
        _loss(checkpointed(checkpointed_input)).backward()

        self.assertIsNotNone(checkpointed_input.grad)
        torch.testing.assert_close(checkpointed_input.grad, plain_input.grad)

    def test_parameter_gradients_match_without_checkpointing(self) -> None:
        """The recomputed backward pass yields the same gradient for every parameter."""
        plain, checkpointed = _build_backbone_pair()
        plain.train()
        checkpointed.train()

        _loss(plain(_input())).backward()
        _loss(checkpointed(_input())).backward()

        plain_gradients = {name: param.grad for name, param in plain.named_parameters()}
        checkpointed_gradients = {
            name: param.grad for name, param in checkpointed.named_parameters()
        }
        self.assertEqual(set(checkpointed_gradients), set(plain_gradients))
        for name, plain_gradient in plain_gradients.items():
            self.assertIsNotNone(plain_gradient, name)
            self.assertIsNotNone(checkpointed_gradients[name], name)
            torch.testing.assert_close(checkpointed_gradients[name], plain_gradient, msg=name)

    def test_batch_norm_statistics_move_once_per_step(self) -> None:
        """
        The recomputation in the backward pass runs every stage again in train mode. Left
        alone, every batch normalization would update its running statistics a second time and
        count two batches per step, doubling the momentum of the model. After one step the
        checkpointed backbone holds the statistics of the plain one.
        """
        plain, checkpointed = _build_backbone_pair()
        plain.train()
        checkpointed.train()

        _loss(plain(_input())).backward()
        _loss(checkpointed(_input())).backward()

        plain_buffers = dict(plain.named_buffers())
        for name, buffer in checkpointed.named_buffers():
            torch.testing.assert_close(buffer, plain_buffers[name], msg=name)
        batch_counts = [
            int(buffer) for name, buffer in checkpointed.named_buffers() if "num_batches" in name
        ]
        self.assertTrue(batch_counts)
        self.assertEqual(set(batch_counts), {1})

    def test_every_stage_is_its_own_checkpoint_segment(self) -> None:
        """Each stage is checkpointed on its own, so one stage's activations live at a time."""
        backbone = _build_backbone(activation_checkpointing=True).train()

        with patch.object(
            second, "checkpoint_block", wraps=activation_checkpointing.checkpoint_block
        ) as checkpoint_call:
            outputs = backbone(_input())

        self.assertEqual(checkpoint_call.call_count, len(backbone.blocks))
        self.assertEqual(
            [call.args[0] for call in checkpoint_call.call_args_list], list(backbone.blocks)
        )
        self.assertEqual(len(outputs), len(_OUT_CHANNELS))

    def test_checkpoint_is_not_used_when_disabled(self) -> None:
        """The default forward never touches the checkpoint utility."""
        backbone = _build_backbone(activation_checkpointing=False).train()

        with patch.object(
            second, "checkpoint_block", wraps=activation_checkpointing.checkpoint_block
        ) as checkpoint_call:
            backbone(_input())

        self.assertEqual(checkpoint_call.call_count, 0)

    def test_checkpoint_is_not_used_without_a_backward_pass(self) -> None:
        """Evaluation and inference run the stages directly, there is nothing to recompute."""
        backbone = _build_backbone(activation_checkpointing=True)

        with patch.object(
            second, "checkpoint_block", wraps=activation_checkpointing.checkpoint_block
        ) as checkpoint_call:
            backbone.eval()
            backbone(_input())
            backbone.train()
            with torch.no_grad():
                backbone(_input())

        self.assertEqual(checkpoint_call.call_count, 0)


if __name__ == "__main__":
    unittest.main()
