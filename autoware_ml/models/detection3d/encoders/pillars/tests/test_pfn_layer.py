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

"""Unit tests for PFN layer."""

import unittest

import torch

from autoware_ml.models.detection3d.encoders.pillars.pfn_layer import PFNLayer


class TestPFNLayer(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PFNLayer instance for all tests. Note that this class will
        be called in each test case.
        """
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # (num_pillars, num_points, num_channels)
        self.input_features = torch.tensor(
            [
                [
                    [1.25, 0.9, -0.5, 7.0, 0.05],
                    [1.50, 2.0, -0.2, 9.0, 0.02],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def test_forward_with_last_layer(self) -> None:
        """Test the forward function of PFNLayer by setting last_layer to True."""
        pfn_layer = PFNLayer(
            in_channels=5,
            out_channels=8,
            last_layer=True,
        ).to(self.device)
        outputs = pfn_layer(
            inputs=self.input_features,
        )
        self.assertEqual(outputs.shape, (2, 1, 8))

        # (num_pillars, 1, num_channels)
        expected_outputs = torch.tensor(
            [
                [[0.0000, 0.0000, 0.0000, 0.0000, 1.2834, 0.0000, 0.0000, 1.2013]],
                [[0.9865, 0.9709, 0.9731, 0.9945, 0.0000, 0.9931, 0.9968, 0.0000]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertTrue(torch.allclose(outputs, expected_outputs, atol=1e-4))

    def test_forward_without_last_layer(self) -> None:
        """Test the forward function of PFNLayer without setting last_layer to True."""
        pfn_layer = PFNLayer(
            in_channels=5,
            out_channels=8,
            last_layer=False,
        ).to(self.device)
        outputs = pfn_layer(
            inputs=self.input_features,
        )
        self.assertEqual(outputs.shape, (2, 2, 8))

        # (num_pillars, num_points, num_output_channels)
        expected_outputs = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.9865, 0.9709, 0.9731, 0.9945, 0.9865, 0.9709, 0.9731, 0.9945],
                    [0.9865, 0.9709, 0.9731, 0.9945, 0.9865, 0.9709, 0.9731, 0.9945],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertTrue(torch.allclose(outputs, expected_outputs, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
