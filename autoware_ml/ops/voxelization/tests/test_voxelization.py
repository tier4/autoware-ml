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

from autoware_ml.ops.voxelization.voxelization import hard_voxelize


class TestHardVoxelizationPointData(unittest.TestCase):
    """Test cases for hard voxelization with actual point data."""

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
        hard_voxelization_outputs = hard_voxelize(
            points=self.raw_points,
            points_batch_indices=self.raw_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )
        self.assertTrue(torch.allclose(hard_voxelization_outputs.voxels, self.expected_voxels))

        # coords is assumed to be in (x, y, z)
        # Concat batch_indices to coords to get (batch_idx, x, y, z)
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


class TestHardVoxelizationDummyPointData(unittest.TestCase):
    """Test cases for hard voxelization with dummy point data."""

    def setUp(self) -> None:
        """
        Set up test data for voxelization tests.
        Note: This method is called before each test method.
        """
        self.voxel_size = torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32)
        self.pc_range = torch.tensor([0.0, 0.0, -2.0, 4.0, 4.0, 2.0], dtype=torch.float32)

    def test_basic_two_voxels(self) -> None:
        """
        Test basic case with two voxels and three points.
        The first two points fall into the same voxel, and the third point falls into a different
        voxel.
        """
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],  # voxel (x=0,y=0,z=0)
                [0.2, 0.2, 0.0, 2.0],  # voxel (x=0,y=0,z=0) - same
                [1.1, 1.1, 0.0, 3.0],  # voxel (x=1,y=1,z=0)
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0, 0, 0], dtype=torch.int32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices,
            self.voxel_size,
            self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )

        voxels = voxels_data.voxels
        coords = voxels_data.coords
        num_points = voxels_data.num_points

        self.assertEqual(voxels.shape, (2, 5, 4))
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(num_points.tolist(), [2, 1])
        # Coords in XYZ order: first voxel is x=0,y=0,z=0; second is x=1,y=1,z=0
        self.assertEqual(coords[0].tolist(), [0, 0, 0])
        self.assertEqual(coords[1].tolist(), [1, 1, 0])

    def test_output_dtypes(self) -> None:
        """Test that the output dtypes of hard_voxelize are as expected."""
        points = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=torch.tensor([0], dtype=torch.int32),
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )
        voxels = voxels_data.voxels
        coords = voxels_data.coords
        num_points = voxels_data.num_points

        self.assertEqual(voxels.dtype, torch.float32)
        self.assertEqual(coords.dtype, torch.int32)
        self.assertEqual(num_points.dtype, torch.int32)

    def test_points_outside_range_filtered(self) -> None:
        """
        Test that points outside the point cloud range are filtered out.
        Expect only the first point to be included in the voxelization result,
        as the other points are outside the defined range.
        """
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0, 1.0],  # inside
                [-1.0, 0.5, 0.0, 2.0],  # x below range
                [5.0, 0.5, 0.0, 3.0],  # x above range
                [0.5, 5.0, 0.0, 4.0],  # y above range
                [0.5, 0.5, 3.0, 5.0],  # z above range
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )

        self.assertEqual(voxels_data.voxels.shape[0], 1)
        self.assertEqual(voxels_data.num_points.tolist(), [1])

    def test_max_num_points_truncation(self) -> None:
        """
        Test that when the number of points in a voxel exceeds max_num_points,
        only first max_num_points are kept and the rest are truncated.
        Expect the first two points to be kept in the voxel, and the last two points
        to be truncated.
        """
        # Four points in the same voxel, max_num_points=2
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],
                [0.2, 0.1, 0.0, 2.0],
                [0.3, 0.1, 0.0, 3.0],
                [0.4, 0.1, 0.0, 4.0],
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=2,
            max_voxels=10,
        )

        self.assertEqual(voxels_data.voxels.shape, (1, 2, 4))
        self.assertEqual(voxels_data.num_points.tolist(), [2])
        # First two points are kept (sorted-key order = input order for same voxel)
        self.assertAlmostEqual(voxels_data.voxels[0, 0, 3].item(), 1.0, places=6)
        self.assertAlmostEqual(voxels_data.voxels[0, 1, 3].item(), 2.0, places=6)

    def test_empty_point_cloud(self) -> None:
        """
        Test that when the input point cloud is empty, the output voxels, coords,
        and num_points are all empty tensors.
        """
        points = torch.zeros((0, 4), dtype=torch.float32)
        points_batch_indices = torch.zeros((0,), dtype=torch.int32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )

        self.assertEqual(voxels_data.voxels.shape[0], 0)
        self.assertEqual(voxels_data.coords.shape[0], 0)
        self.assertEqual(voxels_data.num_points.shape[0], 0)

    def test_all_points_outside_range(self) -> None:
        """
        Test that when all points are outside the point cloud range, the output voxels and
        coords are empty tensors, and num_points is also an empty tensor.
        """
        points = torch.tensor(
            [
                [-5.0, -5.0, 0.0, 1.0],
                [10.0, 10.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0, 1], dtype=torch.int32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )
        self.assertEqual(voxels_data.voxels.shape[0], 0)
        self.assertEqual(voxels_data.coords.shape[0], 0)
        self.assertEqual(voxels_data.num_points.shape[0], 0)

    def test_batch_points_outside_range(self) -> None:
        """
        Test when a sample of points in a batch is fully filtered, only the first and
        third sample of points are included.
        Expect to return voxels shape of (2, max_num_points, num_channels) and
        coords shape of (2, 3).
        """
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],  # Included
                [-5.0, -5.0, 0.0, 2.0],  # Excluded (outside range)
                [0.2, 0.1, 0.0, 3.0],  # Included
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0, 1, 2], dtype=torch.int32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )
        self.assertEqual(voxels_data.voxels.shape, (2, 5, 4))
        self.assertEqual(voxels_data.coords.shape, (2, 3))

    def test_empty_slot_padding_is_zero(self) -> None:
        """
        Test that when a voxel has fewer points than max_num_points, the empty slots
        in the voxel are filled with zeros.
        """
        # One point in one voxel, max_num_points=3  2 empty slots
        points = torch.tensor([[0.5, 0.5, 0.0, 99.0]], dtype=torch.float32)
        points_batch_indices = torch.tensor([0], dtype=torch.int32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=3,
            max_voxels=10,
        )

        self.assertEqual(voxels_data.num_points.tolist(), [1])
        self.assertEqual(voxels_data.voxels[0, 1].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(voxels_data.voxels[0, 2].tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_point_features_preserved(self) -> None:
        """
        Test that the features of points (e.g., intensity) are preserved in the
        voxelization output.
        """
        points = torch.tensor([[0.5, 0.5, 0.0, 7.0, 8.0, 9.0]], dtype=torch.float32)
        points_batch_indices = torch.tensor([0], dtype=torch.int32)
        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=5,
            max_voxels=10,
        )
        self.assertEqual(voxels_data.voxels.shape, (1, 5, 6))
        self.assertAlmostEqual(
            voxels_data.voxels.tolist()[0][0], [0.5, 0.5, 0.0, 7.0, 8.0, 9.0], places=6
        )

    def test_point_raise_misshape_assertion(self) -> None:
        """
        Test that when the row of points and the row of points_batch_indices are not equal,
        an ValueError is raised.
        """
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0, 7.0],
                [0.5, 0.5, 0.0, 7.0],
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0], dtype=torch.int32)

        with self.assertRaises(ValueError):
            hard_voxelize(
                points,
                points_batch_indices=points_batch_indices,
                voxel_size=self.voxel_size,
                point_cloud_range=self.pc_range,
                max_num_points=5,
                max_voxels=10,
            )

    def test_point_voxel_indices_map_every_input_point(self) -> None:
        """Every input point maps to its voxel row; unassigned points get -1."""
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0, 1.0],  # voxel A
                [9.0, 9.0, 0.0, 2.0],  # outside the range
                [0.6, 0.6, 0.0, 3.0],  # voxel A, second slot
                [0.7, 0.7, 0.0, 4.0],  # voxel A, beyond max_num_points
                [1.5, 1.5, 0.0, 5.0],  # voxel B
            ],
            dtype=torch.float32,
        )
        batch_indices = torch.zeros(points.shape[0], dtype=torch.int32)

        result = hard_voxelize(
            points,
            points_batch_indices=batch_indices,
            voxel_size=torch.tensor([1.0, 1.0, 4.0]),
            point_cloud_range=torch.tensor([0.0, 0.0, -2.0, 4.0, 4.0, 2.0]),
            max_num_points=2,
            max_voxels=8,
        )

        indices = result.point_voxel_indices
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(indices.shape, (points.shape[0],))
        self.assertEqual(int(indices[1]), -1)
        self.assertEqual(int(indices[0]), int(indices[2]))
        self.assertEqual(int(indices[0]), int(indices[3]))
        self.assertNotEqual(int(indices[0]), int(indices[4]))
        self.assertEqual(result.num_points[indices[0]].item(), 2)
        self.assertEqual(int(result.num_dropped_voxels), 0)
        self.assertTrue(
            torch.equal(result.coords[indices[4]], torch.tensor([1, 1, 0], dtype=torch.int32))
        )

    def test_point_voxel_indices_mark_dropped_voxels(self) -> None:
        """Points whose voxel exceeds the max_voxels budget are unassigned."""
        points = torch.tensor(
            [[0.5, 0.5, 0.0, 1.0], [1.5, 1.5, 0.0, 2.0], [2.5, 2.5, 0.0, 3.0]],
            dtype=torch.float32,
        )
        result = hard_voxelize(
            points,
            points_batch_indices=torch.zeros(3, dtype=torch.int32),
            voxel_size=torch.tensor([1.0, 1.0, 4.0]),
            point_cloud_range=torch.tensor([0.0, 0.0, -2.0, 4.0, 4.0, 2.0]),
            max_num_points=2,
            max_voxels=2,
        )

        self.assertEqual(result.voxels.shape[0], 2)
        self.assertEqual(int(result.num_dropped_voxels), 1)
        self.assertEqual(int((result.point_voxel_indices < 0).sum()), 1)
        kept = result.point_voxel_indices[result.point_voxel_indices >= 0]
        self.assertEqual(sorted(kept.tolist()), [0, 1])

    def test_boundary_points_never_exceed_grid(self) -> None:
        """Test the points that are on the edge never exceed the grid size."""
        # A point one float32 ulp below the upper range bound can floor to
        # coord == grid_size due to rounding; it must be dropped, not scattered
        # out of bounds downstream.
        voxel_size = torch.tensor([0.24, 0.24, 10.0], dtype=torch.float32)
        pc_range = torch.tensor([-92.16, -92.16, -3.0, 92.16, 92.16, 7.0], dtype=torch.float32)
        edge = torch.nextafter(torch.tensor(92.16), torch.tensor(0.0))
        points = torch.tensor(
            [
                [edge, 0.0, 0.0, 1.0],
                [0.0, edge, 0.0, 2.0],
                [0.0, 0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )
        points_batch_indices = torch.tensor([0, 0, 0], dtype=torch.int32)

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=5,
            max_voxels=10,
        )
        self.assertTrue((voxels_data.coords[:, 1] < 768).all())
        self.assertTrue((voxels_data.coords[:, 2] < 768).all())

    # def test_zyx_coordinate_order(self) -> None:
    #     # Point at x=2, y=3, z=0  grid (2, 3, 0) in XYZ  (0, 3, 2) in ZYX
    #     points = torch.tensor([[2.5, 3.5, 0.0, 1.0]], dtype=torch.float32)

    #     _, coords, _ = hard_voxelize(points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10)

    #     assert coords[0].tolist() == [0, 3, 2]  # z=0, y=3, x=2


if __name__ == "__main__":
    unittest.main()
