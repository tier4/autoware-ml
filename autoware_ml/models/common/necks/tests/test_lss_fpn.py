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

"""Unit tests for the generalized LSS FPN neck."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.necks.lss_fpn import GeneralizedLSSFPN


class TestGeneralizedLSSFPN(unittest.TestCase):
    """Unit tests for the top-down LSS feature pyramid neck."""

    def setUp(self) -> None:
        """Set up a three-level ResNet-18 style pyramid at strides 8, 16 and 32."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = [128, 256, 512]
        self.out_channels = 64
        self.pyramid = (
            torch.randn(self.batch_size, 128, 16, 16, device=self.device),
            torch.randn(self.batch_size, 256, 8, 8, device=self.device),
            torch.randn(self.batch_size, 512, 4, 4, device=self.device),
        )
        self.neck = (
            GeneralizedLSSFPN(in_channels=self.in_channels, out_channels=self.out_channels)
            .to(self.device)
            .eval()
        )

    def test_projects_feature_pyramid_to_one_fewer_level(self) -> None:
        """Test that the neck emits ``len(in_channels) - 1`` maps of ``out_channels`` width."""
        with torch.no_grad():
            outputs = self.neck(self.pyramid)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.out_channels, 16, 16))
        self.assertEqual(outputs[1].shape, (self.batch_size, self.out_channels, 8, 8))
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())

    def test_rejects_single_level_pyramid(self) -> None:
        """Test that the top-down pathway needs at least two feature levels."""
        with self.assertRaisesRegex(ValueError, "at least two feature levels"):
            GeneralizedLSSFPN(in_channels=[128], out_channels=self.out_channels)

    def test_gradients_reach_every_level(self) -> None:
        """Test that every input level contributes to the fused outputs."""
        pyramid = tuple(level.clone().requires_grad_() for level in self.pyramid)

        outputs = self.neck(pyramid)
        sum(output.sum() for output in outputs).backward()

        for level_index, level in enumerate(pyramid):
            with self.subTest(level=level_index):
                self.assertIsNotNone(level.grad)
                assert level.grad is not None
                self.assertGreater(level.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
