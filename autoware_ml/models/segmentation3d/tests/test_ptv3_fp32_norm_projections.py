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

"""The projections that feed a BatchNorm stay in fp32 under autocast."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    FullPrecisionLinear,
    SerializedUnpooling,
)


class TestFullPrecisionLinear(unittest.TestCase):
    """Unit tests for the linear layer that computes in fp32 under autocast."""

    @unittest.skipUnless(torch.cuda.is_available(), "autocast on CUDA")
    def test_projection_survives_inputs_that_overflow_fp16(self) -> None:
        """
        Input: a 64 -> 32 projection with weights scaled by 40 and 1000 features scaled by
        3000, run under fp16 autocast both as FullPrecisionLinear and as an nn.Linear with
        the same weights.
        Expected: the products exceed the fp16 range (65,504), so the plain layer overflows,
        while FullPrecisionLinear computes in fp32 and stays finite.
        Check: the plain output has non-finite values; the FullPrecisionLinear output is fp32,
        finite and matches the fp32 projection of the same input.
        """
        torch.manual_seed(0)
        layer = FullPrecisionLinear(64, 32).cuda()
        with torch.no_grad():
            layer.weight.mul_(40.0)
        linear = nn.Linear(64, 32).cuda()
        linear.load_state_dict(layer.state_dict())
        feat = torch.randn(1000, 64, device="cuda") * 3000.0  # products exceed 65,504
        with torch.autocast("cuda", dtype=torch.float16):
            naive = linear(feat)
            guarded = layer(feat)
        self.assertFalse(
            torch.isfinite(naive).all(), "the test input must overflow the plain fp16 path"
        )
        self.assertEqual(guarded.dtype, torch.float32)
        self.assertTrue(torch.isfinite(guarded).all())
        torch.testing.assert_close(guarded, linear(feat.float()), rtol=1e-4, atol=1e-2)

    def test_layer_is_a_plain_linear_outside_autocast(self) -> None:
        """
        Input: an 8 -> 4 FullPrecisionLinear and an nn.Linear loaded with its state dict,
        applied to five random features without autocast.
        Expected: outside autocast FullPrecisionLinear is a plain linear layer with the same
        parameters, so checkpoints load unchanged.
        Check: both outputs match, and the state dict holds only ``weight`` and ``bias``.
        """
        torch.manual_seed(1)
        layer = FullPrecisionLinear(8, 4)
        plain = nn.Linear(8, 4)
        plain.load_state_dict(layer.state_dict())
        x = torch.randn(5, 8)
        torch.testing.assert_close(layer(x), plain(x))
        self.assertEqual(set(layer.state_dict()), {"weight", "bias"})


class TestSerializedUnpooling(unittest.TestCase):
    """Unit tests for the unpooling projections."""

    def test_unpooling_state_dict_layout_is_unchanged(self) -> None:
        """
        Input: a SerializedUnpooling with 16 input, 8 skip and 12 output channels.
        Expected: the projections are FullPrecisionLinear inside the same PointSequential, so
        the parameter names of existing checkpoints do not change.
        Check: the state dict has ``proj.0.weight``, ``proj_skip.0.weight`` and the
        BatchNorm running means at ``proj.1`` and ``proj_skip.1``.
        """
        module = SerializedUnpooling(16, 8, 12)
        keys = set(module.state_dict())
        self.assertIn("proj.0.weight", keys)
        self.assertIn("proj_skip.0.weight", keys)
        self.assertIn("proj.1.running_mean", keys)
        self.assertIn("proj_skip.1.running_mean", keys)


if __name__ == "__main__":
    unittest.main()
