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

import torch

from autoware_ml.preprocessing.detection3d.point_pillar import PointPillarPreprocessor


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
