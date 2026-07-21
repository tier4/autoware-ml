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

"""Test cases for voxelization operations."""

from pathlib import Path
import unittest

import numpy as np
import torch

from autoware_ml.ops.voxelization.voxelization import batch_hard_voxelize


class TestVoxelization(unittest.TestCase):
    """Test cases for voxelization operations."""

    @classmethod
    def setUpClass(cls):
        """
        Set up test data for voxelization tests.
        Note: This method is called
        once for the entire test class.
        """

        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.points_file = cls.test_data_dir / "raw_points.npz"
        cls.raw_batch_indices_file = cls.test_data_dir / "raw_batch_indices.npz"
        cls.expected_voxels_file = cls.test_data_dir / "expected_voxels.npz"
        cls.expected_coords_file = cls.test_data_dir / "expected_coords.npz"
        cls.expected_num_points_file = cls.test_data_dir / "expected_num_points.npz"
        cls.npz_array_name = "arr_0"  # Default name for a single array in .npz files

        # Use GPU when CUDA is available, otherwise fall back to CPU
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load raw points and points-level batch indices, and move them to the
        # appropriate device (CPU or GPU)
        cls.raw_points = torch.tensor(
            np.load(cls.points_file)[cls.npz_array_name], device=cls.device
        )
        cls.raw_batch_indices = torch.tensor(
            np.load(cls.raw_batch_indices_file)[cls.npz_array_name], device=cls.device
        )

        # Load expected outputs, and move them to the appropriate device (CPU or GPU)
        cls.expected_voxels = torch.tensor(
            np.load(cls.expected_voxels_file)[cls.npz_array_name], device=cls.device
        )
        cls.expected_coords = torch.tensor(
            np.load(cls.expected_coords_file)[cls.npz_array_name], device=cls.device
        )
        cls.expected_num_points = torch.tensor(
            np.load(cls.expected_num_points_file)[cls.npz_array_name], device=cls.device
        )

        # Configs
        cls.point_cloud_range = torch.tensor(
            [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0], device=cls.device, dtype=torch.float32
        )
        cls.voxel_size = torch.tensor([0.24, 0.24, 8.0], device=cls.device)
        cls.max_num_points = 32
        cls.max_voxels = 96000

    def test_batch_hard_voxelize(self):
        """
        Test batch_hard_voxelize function to ensure it returns the expected voxels, coords,
        num_points_file, where each expected result is generated from sample-level hard_voxelize
        function to ensure bit-identical results.
        """
        hard_voxelization_outputs = batch_hard_voxelize(
            points=self.raw_points,
            points_batch_indices=self.raw_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )
        self.assertTrue(torch.allclose(hard_voxelization_outputs.voxels, self.expected_voxels))

        # coords is assumed to be in (x, y, z)
        # Concat batch_indices to coords to get (batch, x, y, z)
        voxel_coords = torch.cat(
            [
                hard_voxelization_outputs.batch_indices.unsqueeze(1),
                hard_voxelization_outputs.coords,
            ],
            dim=1,
        )
        self.assertTrue(torch.allclose(voxel_coords, self.expected_coords))
        self.assertTrue(
            torch.allclose(hard_voxelization_outputs.num_points, self.expected_num_points)
        )


if __name__ == "__main__":
    unittest.main()
