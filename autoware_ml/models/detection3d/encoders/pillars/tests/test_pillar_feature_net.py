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

"""Unit tests for PillarFeatureNet."""

import unittest

import torch

from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class TestPillarFeatureNet(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PointPillarPreprocessor instance for all tests. Note that this class will
        be called in each test case.
        """
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pts_voxel_encoder = PillarFeatureNet(
            in_channels=5,
            feat_channels=[8],
            voxel_size=[0.5, 0.5, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        ).to(self.device)

    def test_encode(self) -> None:
        """Test the decorate function of PillarFeatureNet."""
        voxels_data = VoxelsData(
            voxels=torch.tensor(
                [[[1.25, 0.9, -0.5, 7.0, 0.05]]], dtype=torch.float32, device=self.device
            ),
            num_points=torch.tensor([1], dtype=torch.int32, device=self.device),
            coords=torch.tensor([[2, 1, 0]], dtype=torch.int32, device=self.device),  # (x, y, z)
            batch_indices=torch.tensor([0], dtype=torch.int32, device=self.device),
            point_voxel_indices=torch.zeros((0,), dtype=torch.int64),
            num_dropped_voxels=torch.zeros((), dtype=torch.int64),
        )
        features = self.pts_voxel_encoder.encode(
            voxels_data=voxels_data,
        )

        expected = torch.tensor(
            [1.25, 0.9, -0.5, 7.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.15, -0.5], device=self.device
        )
        self.assertTrue(torch.allclose(features.squeeze(1)[0], expected, atol=1e-6))

    def test_forward(self) -> None:
        """Test the forward function of PillarFeatureNet."""
        voxels_data = VoxelsData(
            voxels=torch.tensor(
                [
                    [
                        [1.25, 0.9, -0.5, 7.0, 0.05],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.3, 0.4, 0.5],
                    ],
                    [
                        [1.30, 0.7, 0.2, 0.1, 0.2],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],  # (2, 3, 11)
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            num_points=torch.tensor([1, 2], dtype=torch.int32, device=self.device),
            coords=torch.tensor(
                [[2, 1, 0], [2, 1, 0]], dtype=torch.int32, device=self.device
            ),  # (x, y, z)
            batch_indices=torch.tensor([0, 0], dtype=torch.int32, device=self.device),
            point_voxel_indices=torch.zeros((0,), dtype=torch.int64),
            num_dropped_voxels=torch.zeros((), dtype=torch.int64),
        )
        features = self.pts_voxel_encoder(voxels_data)
        self.assertEqual(features.shape, (2, 8))  # (num_voxels, feat_channels)
        expected_outputs = torch.tensor(
            [
                [0.4582, 0.5374, 1.9813, 0.5429, 1.9834, 0.6387, 2.1574, 0.5094],
                [0.5929, 0.5374, 0.3992, 0.5730, 0.0000, 0.6387, 0.0000, 0.5764],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertTrue(torch.allclose(features, expected_outputs, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
