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

"""Unit tests for the SECOND FPN neck: output shapes and the activation checkpointing path."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from autoware_ml.models.detection3d.necks import second_fpn
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN

_IN_CHANNELS = (8, 16)
_OUT_CHANNELS = (8, 8)
_UPSAMPLE_STRIDES = (1, 2)
_BATCH_SIZE = 2
_BEV_SIZE = 16


def _build_neck(activation_checkpointing: bool) -> SECONDFPN:
    """Build a two-stage upsampling neck from a fixed seed."""
    torch.manual_seed(0)
    return SECONDFPN(
        in_channels=_IN_CHANNELS,
        out_channels=_OUT_CHANNELS,
        upsample_strides=_UPSAMPLE_STRIDES,
        activation_checkpointing=activation_checkpointing,
    )


def _build_neck_pair() -> tuple[SECONDFPN, SECONDFPN]:
    """Build a plain and a checkpointed neck sharing the same weights."""
    plain = _build_neck(activation_checkpointing=False)
    checkpointed = _build_neck(activation_checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    return plain, checkpointed


def _inputs() -> list[torch.Tensor]:
    """Build fixed backbone feature maps, high resolution first."""
    torch.manual_seed(1)
    return [
        torch.randn(_BATCH_SIZE, _IN_CHANNELS[0], _BEV_SIZE, _BEV_SIZE),
        torch.randn(_BATCH_SIZE, _IN_CHANNELS[1], _BEV_SIZE // 2, _BEV_SIZE // 2),
    ]


class TestSECONDFPN(unittest.TestCase):
    """Unit tests for the SECOND FPN forward pass."""

    def test_upsampled_stages_are_concatenated_along_channels(self) -> None:
        """Every stage is brought to the first stage resolution and stacked on the channel axis."""
        neck = _build_neck(activation_checkpointing=False).eval()

        with torch.no_grad():
            fused = neck(_inputs())

        self.assertEqual(fused.shape, (_BATCH_SIZE, sum(_OUT_CHANNELS), _BEV_SIZE, _BEV_SIZE))

    def test_fractional_stride_downsamples_with_a_strided_convolution(self) -> None:
        """A stride below one builds a strided convolution instead of a transposed one."""
        torch.manual_seed(0)
        neck = SECONDFPN(in_channels=(8, 16), out_channels=(4, 4), upsample_strides=(0.5, 1)).eval()

        with torch.no_grad():
            fused = neck(_inputs())

        self.assertIsInstance(neck.blocks[0], torch.nn.Sequential)
        self.assertEqual(fused.shape, (_BATCH_SIZE, 8, _BEV_SIZE // 2, _BEV_SIZE // 2))


class TestActivationCheckpointing(unittest.TestCase):
    """Activation checkpointing trades memory for recomputation without changing the math."""

    def test_eval_outputs_match_without_checkpointing(self) -> None:
        """In eval mode the checkpointed output equals the plain output."""
        plain, checkpointed = _build_neck_pair()
        plain.eval()
        checkpointed.eval()

        with torch.no_grad():
            plain_output = plain(_inputs())
            checkpointed_output = checkpointed(_inputs())

        torch.testing.assert_close(checkpointed_output, plain_output)

    def test_train_outputs_match_without_checkpointing(self) -> None:
        """In train mode batch statistics drive the normalization and the outputs still match."""
        plain, checkpointed = _build_neck_pair()
        plain.train()
        checkpointed.train()

        plain_output = plain(_inputs())
        checkpointed_output = checkpointed(_inputs())

        torch.testing.assert_close(checkpointed_output, plain_output)

    def test_downsampling_stage_matches_without_checkpointing(self) -> None:
        """The strided-convolution branch is checkpointed the same way as the transposed one."""
        torch.manual_seed(0)
        plain = SECONDFPN(in_channels=(8, 16), out_channels=(4, 4), upsample_strides=(0.5, 1))
        checkpointed = SECONDFPN(
            in_channels=(8, 16),
            out_channels=(4, 4),
            upsample_strides=(0.5, 1),
            activation_checkpointing=True,
        )
        checkpointed.load_state_dict(plain.state_dict())
        plain.train()
        checkpointed.train()

        torch.testing.assert_close(checkpointed(_inputs()), plain(_inputs()))

    def test_input_gradients_match_without_checkpointing(self) -> None:
        """The recomputed backward pass yields the same gradient for every input stage."""
        plain, checkpointed = _build_neck_pair()
        plain.train()
        checkpointed.train()
        plain_inputs = [feature.requires_grad_(True) for feature in _inputs()]
        checkpointed_inputs = [feature.requires_grad_(True) for feature in _inputs()]

        plain(plain_inputs).sum().backward()
        checkpointed(checkpointed_inputs).sum().backward()

        for plain_input, checkpointed_input in zip(plain_inputs, checkpointed_inputs):
            self.assertIsNotNone(checkpointed_input.grad)
            torch.testing.assert_close(checkpointed_input.grad, plain_input.grad)

    def test_parameter_gradients_match_without_checkpointing(self) -> None:
        """The recomputed backward pass yields the same gradient for every parameter."""
        plain, checkpointed = _build_neck_pair()
        plain.train()
        checkpointed.train()

        plain(_inputs()).sum().backward()
        checkpointed(_inputs()).sum().backward()

        plain_gradients = {name: param.grad for name, param in plain.named_parameters()}
        checkpointed_gradients = {
            name: param.grad for name, param in checkpointed.named_parameters()
        }
        self.assertEqual(set(checkpointed_gradients), set(plain_gradients))
        for name, plain_gradient in plain_gradients.items():
            self.assertIsNotNone(plain_gradient, name)
            self.assertIsNotNone(checkpointed_gradients[name], name)
            torch.testing.assert_close(checkpointed_gradients[name], plain_gradient, msg=name)

    def test_checkpoint_is_used_when_enabled(self) -> None:
        """The whole forward is wrapped in one non-reentrant checkpoint call."""
        neck = _build_neck(activation_checkpointing=True).train()

        with patch.object(second_fpn, "checkpoint", wraps=torch_checkpoint) as checkpoint_call:
            fused = neck(_inputs())

        self.assertEqual(checkpoint_call.call_count, 1)
        self.assertEqual(checkpoint_call.call_args.args[0], neck.custom_forward)
        self.assertFalse(checkpoint_call.call_args.kwargs["use_reentrant"])
        self.assertEqual(fused.shape, (_BATCH_SIZE, sum(_OUT_CHANNELS), _BEV_SIZE, _BEV_SIZE))

    def test_checkpoint_is_not_used_when_disabled(self) -> None:
        """The default forward never touches the checkpoint utility."""
        neck = _build_neck(activation_checkpointing=False).train()

        with patch.object(second_fpn, "checkpoint", wraps=torch_checkpoint) as checkpoint_call:
            neck(_inputs())

        self.assertEqual(checkpoint_call.call_count, 0)


if __name__ == "__main__":
    unittest.main()
