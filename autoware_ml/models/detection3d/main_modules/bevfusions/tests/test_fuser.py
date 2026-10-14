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

"""Unit tests for the BEV feature fusion modules."""

from __future__ import annotations

import unittest

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.main_modules.bevfusions.fuser import ConvFuser


class TestConvFuser(unittest.TestCase):
    """Unit tests for the ConvFuser."""

    def setUp(self) -> None:
        """Set up the device, the branch layout and a fuser in eval mode so BatchNorm is deterministic."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = [4, 6]
        self.out_channels = 8
        self.bev_shape = (8, 8)
        self.fuser = (
            ConvFuser(in_channels=self.in_channels, out_channels=self.out_channels)
            .to(self.device)
            .eval()
        )
        self.bev_features = self._build_bev_features(self.in_channels)

    def _build_bev_features(
        self, in_channels: list[int]
    ) -> list[Float32[torch.Tensor, "batch_size channels height width"]]:
        """Build one random BEV feature map per branch with the given channel counts."""
        return [
            torch.randn(self.batch_size, channels, *self.bev_shape, device=self.device)
            for channels in in_channels
        ]

    def _fuse(
        self,
        fuser: ConvFuser,
        bev_features: list[Float32[torch.Tensor, "batch_size channels height width"]],
    ) -> Float32[torch.Tensor, "batch_size out_channels height width"]:
        """Run the fuser without tracking gradients."""
        with torch.no_grad():
            return fuser(bev_features)

    def test_projection_layers_follow_the_channel_layout(self) -> None:
        """Test that the fuser projects the concatenated channels with a bias-free convolution."""
        conv, norm, activation = self.fuser.proj

        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual(conv.in_channels, sum(self.in_channels))
        self.assertEqual(conv.out_channels, self.out_channels)
        self.assertIsNone(conv.bias)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.padding, (1, 1))
        self.assertIsInstance(norm, nn.BatchNorm2d)
        self.assertEqual(norm.num_features, self.out_channels)
        self.assertIsInstance(activation, nn.ReLU)

    def test_forward_fuses_branches_into_out_channels(self) -> None:
        """Test that the fused map has the requested channels and keeps the BEV resolution."""
        fused = self._fuse(self.fuser, self.bev_features)

        self.assertEqual(fused.shape, (self.batch_size, self.out_channels, *self.bev_shape))
        self.assertTrue(torch.isfinite(fused).all())
        # The fusion ends in a ReLU.
        self.assertTrue(torch.all(fused >= 0.0))
        self.assertGreater(fused.sum().item(), 0.0)

    def test_forward_equals_projection_of_channel_concatenation(self) -> None:
        """Test that fusion is exactly the projection applied to the channel-wise concatenation."""
        fused = self._fuse(self.fuser, self.bev_features)

        with torch.no_grad():
            expected = self.fuser.proj(torch.cat(self.bev_features, dim=1))

        torch.testing.assert_close(fused, expected)

    def test_forward_depends_on_branch_order(self) -> None:
        """Test that the branches are concatenated in the order given, so swapping them matters."""
        fuser = ConvFuser(in_channels=[4, 4], out_channels=self.out_channels).to(self.device).eval()
        first, second = self._build_bev_features([4, 4])

        fused = self._fuse(fuser, [first, second])
        swapped = self._fuse(fuser, [second, first])

        self.assertFalse(torch.allclose(fused, swapped))

    def test_forward_accepts_more_than_two_branches(self) -> None:
        """Test that any number of branches above one is fused."""
        in_channels = [2, 3, 5]
        fuser = (
            ConvFuser(in_channels=in_channels, out_channels=self.out_channels)
            .to(self.device)
            .eval()
        )

        fused = self._fuse(fuser, self._build_bev_features(in_channels))

        self.assertEqual(fused.shape, (self.batch_size, self.out_channels, *self.bev_shape))

    def test_forward_rejects_single_branch(self) -> None:
        """Test that fusing a single branch is rejected, since there is nothing to fuse."""
        with self.assertRaises(AssertionError):
            self.fuser([self.bev_features[0]])

    def test_forward_rejects_mismatched_spatial_shapes(self) -> None:
        """Test that branches on different BEV grids cannot be concatenated."""
        height, width = self.bev_shape
        smaller = torch.randn(
            self.batch_size, self.in_channels[1], height // 2, width // 2, device=self.device
        )

        with self.assertRaises(RuntimeError):
            self.fuser([self.bev_features[0], smaller])

    def test_kernel_size_and_padding_control_the_output_resolution(self) -> None:
        """Test that a matching padding keeps the resolution and no padding shrinks it."""
        height, width = self.bev_shape
        for kernel_size, padding, expected_shape in (
            (1, 0, (height, width)),
            (5, 2, (height, width)),
            (3, 0, (height - 2, width - 2)),
        ):
            with self.subTest(kernel_size=kernel_size, padding=padding):
                fuser = (
                    ConvFuser(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                    .to(self.device)
                    .eval()
                )

                fused = self._fuse(fuser, self.bev_features)

                self.assertEqual(fused.shape, (self.batch_size, self.out_channels, *expected_shape))

    def test_gradients_reach_every_branch(self) -> None:
        """Test that the fused map is differentiable with respect to every input branch."""
        bev_features = [
            features.requires_grad_() for features in self._build_bev_features(self.in_channels)
        ]

        self.fuser(bev_features).sum().backward()

        for branch_index, features in enumerate(bev_features):
            with self.subTest(branch=branch_index):
                self.assertIsNotNone(features.grad)
                assert features.grad is not None
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
