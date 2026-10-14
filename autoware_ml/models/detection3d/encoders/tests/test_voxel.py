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

"""Unit tests for the voxel feature encoders."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.encoders.voxel import HardSimpleVoxelSinCosEncoder


class TestHardSimpleVoxelSinCosEncoder(unittest.TestCase):
    """Unit tests for the mean-pooling sin/cos Fourier voxel encoder."""

    def setUp(self) -> None:
        """Set up an encoder over ``(x, y, z, intensity)`` with the 120 m TransFusion range."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = 4
        self.min_norm_values = [-122.4, -122.4, -3.0, 0.0]
        self.max_norm_values = [122.4, 122.4, 5.0, 255.0]
        self.encoder = HardSimpleVoxelSinCosEncoder(
            self.min_norm_values, self.max_norm_values, in_channels=self.in_channels
        ).to(self.device)

    def _coords(self, num_voxels: int) -> torch.Tensor:
        """Build placeholder ``[batch, z, y, x]`` coordinates; the encoder ignores them."""
        return torch.zeros(num_voxels, 4, dtype=torch.int64, device=self.device)

    def test_output_dim_and_sincos_identity(self) -> None:
        """
        Test that the features are ``2 * C ** 2`` wide and that every cos/sin pair lies on the
        unit circle.
        """
        num_voxels, max_points = 9, 32
        voxels = torch.randn(num_voxels, max_points, self.in_channels, device=self.device)
        num_points = torch.randint(1, max_points, (num_voxels,), device=self.device)

        out = self.encoder(voxels, num_points, self._coords(num_voxels))

        num_frequencies = self.in_channels * self.in_channels
        self.assertEqual(out.shape, (num_voxels, 2 * num_frequencies))
        cos_part, sin_part = out[:, :num_frequencies], out[:, num_frequencies:]
        torch.testing.assert_close(
            cos_part**2 + sin_part**2, torch.ones_like(cos_part), atol=1e-5, rtol=0.0
        )

    def test_mean_pooling_ignores_padding(self) -> None:
        """Test that padded point slots do not change the pooled feature of a voxel."""
        point = torch.tensor([10.0, 20.0, 1.0, 100.0], device=self.device)
        padded = torch.zeros(1, 4, self.in_channels, device=self.device)
        padded[0, 0] = point  # one real point, the rest is padding
        single = point.view(1, 1, self.in_channels)
        num_points = torch.tensor([1], device=self.device)

        out_padded = self.encoder(padded, num_points, self._coords(1))
        out_single = self.encoder(single, num_points, self._coords(1))

        torch.testing.assert_close(out_padded, out_single, atol=1e-5, rtol=0.0)

    def test_mean_pooling_averages_valid_points(self) -> None:
        """Test that a voxel of two points encodes like a single point at their mean."""
        points = torch.tensor([[0.0, 4.0, -1.0, 40.0], [2.0, 8.0, 3.0, 80.0]], device=self.device)
        pair = points.view(1, 2, self.in_channels)
        mean = points.mean(dim=0).view(1, 1, self.in_channels)

        out_pair = self.encoder(pair, torch.tensor([2], device=self.device), self._coords(1))
        out_mean = self.encoder(mean, torch.tensor([1], device=self.device), self._coords(1))

        torch.testing.assert_close(out_pair, out_mean, atol=1e-5, rtol=0.0)

    def test_rejects_norm_values_of_the_wrong_length(self) -> None:
        """Test that the per-channel normalization bounds must match ``in_channels``."""
        with self.assertRaisesRegex(ValueError, "in_channels"):
            HardSimpleVoxelSinCosEncoder([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], in_channels=4)


if __name__ == "__main__":
    unittest.main()
