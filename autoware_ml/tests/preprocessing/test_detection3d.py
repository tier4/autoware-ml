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

"""Unit tests for detection preprocessing and voxelization."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.ops.voxelization import hard_voxelize
from autoware_ml.preprocessing.detection3d.point_pillar import PointPillarPreprocessor

VOXEL_SIZE = torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32)
PC_RANGE = torch.tensor([0.0, 0.0, -2.0, 4.0, 4.0, 2.0], dtype=torch.float32)


class TestHardVoxelize:
    def test_basic_two_voxels(self) -> None:
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],  # voxel (x=0,y=0,z=0)
                [0.2, 0.2, 0.0, 2.0],  # voxel (x=0,y=0,z=0) - same
                [1.1, 1.1, 0.0, 3.0],  # voxel (x=1,y=1,z=0)
            ],
            dtype=torch.float32,
        )

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10
        )

        assert voxels.shape == (2, 5, 4)
        assert coords.shape == (2, 3)
        assert num_points.tolist() == [2, 1]
        # Coords in ZYX order: first voxel is z=0,y=0,x=0; second is z=0,y=1,x=1
        assert coords[0].tolist() == [0, 0, 0]
        assert coords[1].tolist() == [0, 1, 1]

    def test_output_dtypes(self) -> None:
        points = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10
        )

        assert voxels.dtype == torch.float32
        assert coords.dtype == torch.int32
        assert num_points.dtype == torch.int32

    def test_points_outside_range_filtered(self) -> None:
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

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10
        )

        assert voxels.shape[0] == 1
        assert num_points.tolist() == [1]

    def test_max_num_points_truncation(self) -> None:
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

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=2, max_voxels=10
        )

        assert voxels.shape == (1, 2, 4)
        assert num_points.tolist() == [2]
        # First two points are kept (sorted-key order = input order for same voxel)
        assert voxels[0, 0, 3].item() == pytest.approx(1.0)
        assert voxels[0, 1, 3].item() == pytest.approx(2.0)

    def test_max_voxels_truncation(self) -> None:
        # Three distinct voxels, max_voxels=2
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0, 1.0],  # voxel key 0
                [1.5, 0.5, 0.0, 2.0],  # voxel key 1
                [2.5, 0.5, 0.0, 3.0],  # voxel key 2
            ],
            dtype=torch.float32,
        )

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=2
        )

        assert voxels.shape[0] == 2
        assert coords.shape[0] == 2
        assert num_points.shape[0] == 2

    def test_empty_point_cloud(self) -> None:
        points = torch.zeros((0, 4), dtype=torch.float32)

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10
        )

        assert voxels.shape[0] == 0
        assert coords.shape[0] == 0
        assert num_points.shape[0] == 0

    def test_all_points_outside_range(self) -> None:
        points = torch.tensor(
            [
                [-5.0, -5.0, 0.0, 1.0],
                [10.0, 10.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        )

        voxels, coords, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10
        )

        assert voxels.shape[0] == 0
        assert coords.shape[0] == 0

    def test_empty_slot_padding_is_zero(self) -> None:
        # One point in one voxel, max_num_points=3  2 empty slots
        points = torch.tensor([[0.5, 0.5, 0.0, 99.0]], dtype=torch.float32)

        voxels, _, num_points = hard_voxelize(
            points, VOXEL_SIZE, PC_RANGE, max_num_points=3, max_voxels=10
        )

        assert num_points.tolist() == [1]
        assert voxels[0, 1].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert voxels[0, 2].tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_point_features_preserved(self) -> None:
        points = torch.tensor([[0.5, 0.5, 0.0, 7.0, 8.0, 9.0]], dtype=torch.float32)

        voxels, _, _ = hard_voxelize(points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10)

        assert voxels[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.0, 7.0, 8.0, 9.0])

    def test_boundary_points_never_exceed_grid(self) -> None:
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

        _, coords, _ = hard_voxelize(points, voxel_size, pc_range, max_num_points=5, max_voxels=10)

        assert (coords[:, 1] < 768).all()
        assert (coords[:, 2] < 768).all()

    def test_zyx_coordinate_order(self) -> None:
        # Point at x=2, y=3, z=0  grid (2, 3, 0) in XYZ  (0, 3, 2) in ZYX
        points = torch.tensor([[2.5, 3.5, 0.0, 1.0]], dtype=torch.float32)

        _, coords, _ = hard_voxelize(points, VOXEL_SIZE, PC_RANGE, max_num_points=5, max_voxels=10)

        assert coords[0].tolist() == [0, 3, 2]  # z=0, y=3, x=2


class TestPointPillarPreprocessor:
    def test_forward_builds_padded_pillars(self) -> None:
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=8,
        )
        batch = {
            "points": [
                torch.tensor(
                    [
                        [0.1, 0.1, 0.0, 1.0],
                        [0.2, 0.2, 0.0, 2.0],
                        [1.1, 1.1, 0.0, 3.0],
                    ],
                    dtype=torch.float32,
                )
            ]
        }

        outputs = preprocessor(batch)

        assert outputs["voxels"].shape == (2, 2, 4)
        assert outputs["num_points"].tolist() == [2, 1]
        assert outputs["voxel_coords"].shape == (2, 4)
        assert outputs["voxel_coords"][:, 0].tolist() == [0, 0]

    def test_batch_column_increments_per_sample(self) -> None:
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=5,
            max_voxels=10,
        )
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        batch = {"points": [point, point, point]}

        outputs = preprocessor(batch)

        assert outputs["voxel_coords"][:, 0].tolist() == [0, 1, 2]

    def test_empty_sample_in_batch(self) -> None:
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=5,
            max_voxels=10,
        )
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        empty = torch.zeros((0, 4), dtype=torch.float32)
        batch = {"points": [point, empty, point]}

        outputs = preprocessor(batch)

        # Two non-empty samples  2 voxels total
        assert outputs["voxels"].shape[0] == 2
        assert set(outputs["voxel_coords"][:, 0].tolist()) == {0, 2}

    def test_empty_batch_returns_empty_pillar_tensors(self) -> None:
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=5,
            max_voxels=10,
        )

        outputs = preprocessor({"points": []})

        assert outputs["voxels"].shape == (0, 5, 0)
        assert outputs["num_points"].shape == (0,)
        assert outputs["voxel_coords"].shape == (0, 4)

    def test_passthrough_of_existing_keys(self) -> None:
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=5,
            max_voxels=10,
        )
        sentinel = torch.tensor([42.0])
        batch = {
            "points": [torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)],
            "gt_boxes": sentinel,
        }

        outputs = preprocessor(batch)

        assert outputs["gt_boxes"] is sentinel
