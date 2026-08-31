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

"""Unit tests for point pillar preprocessing."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.preprocessing.detection3d.point_pillar import PointPillarPreprocessor


class TestPointPillarPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PointPillarPreprocessor instance for all tests. Note that this class will
        be called in each test case.
        """
        self.point_pillar_preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=8,
            voxelization_z_order_first=True,  # This is used for backward-compatible, and will be removed very soon.
        )
        torch.manual_seed(0)

    def test_forward_builds_padded_pillars(self) -> None:
        """
        Test that the forward method correctly builds padded pillars from a batch of point
        clouds.
        """
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

        outputs = self.point_pillar_preprocessor(batch, is_training=True)
        self.assertEqual(outputs["voxels"].shape, (2, 2, 4))
        self.assertEqual(outputs["num_points"].tolist(), [2, 1])
        self.assertEqual(outputs["voxel_coords"].shape, (2, 4))
        self.assertEqual(outputs["voxel_coords"][:, 0].tolist(), [0, 0])

    def test_point_voxel_indices_follow_concatenated_batch_order(self) -> None:
        """Point indices are emitted for the batch-concatenated points, -1 when unassigned."""
        batch = {
            "points": [
                torch.tensor([[0.1, 0.1, 0.0, 1.0], [9.0, 9.0, 0.0, 2.0]], dtype=torch.float32),
                torch.tensor([[0.2, 0.2, 0.0, 3.0], [1.1, 1.1, 0.0, 4.0]], dtype=torch.float32),
            ]
        }

        outputs = self.point_pillar_preprocessor(batch, is_training=True)

        indices = outputs["point_voxel_indices"]
        self.assertEqual(indices.shape, (4,))
        self.assertEqual(int(indices[1]), -1)
        assigned = indices[indices >= 0]
        self.assertEqual(sorted(assigned.tolist()), [0, 1, 2])
        self.assertEqual(int(outputs["num_dropped_voxels"]), 0)
        # the row a point maps to carries that point's batch index
        self.assertEqual(int(outputs["voxel_coords"][indices[0], 0]), 0)
        self.assertEqual(int(outputs["voxel_coords"][indices[2], 0]), 1)
        self.assertEqual(int(outputs["voxel_coords"][indices[3], 0]), 1)

    def test_batch_column_increments_per_sample(self) -> None:
        """
        Test that the batch column in voxel coordinates increments correctly
        for each sample in the batch.
        """
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        batch = {"points": [point, point, point]}

        outputs = self.point_pillar_preprocessor(batch, is_training=True)

        self.assertEqual(outputs["voxel_coords"][:, 0].tolist(), [0, 1, 2])

    def test_empty_sample_in_batch(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly handles a batch containing an empty
        sample.
        """
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        empty = torch.zeros((0, 4), dtype=torch.float32)
        batch = {"points": [point, empty, point]}

        outputs = self.point_pillar_preprocessor(batch, is_training=True)

        # Two non-empty samples  2 voxels total
        self.assertEqual(outputs["voxels"].shape[0], 2)
        self.assertEqual(set(outputs["voxel_coords"][:, 0].tolist()), {0, 2})

    def test_empty_batch_returns_empty_pillar_tensors(self) -> None:
        """
        Test that the PointPillarPreprocessor returns empty pillar tensors when given an
        empty batch.
        """
        outputs = self.point_pillar_preprocessor({"points": []}, is_training=True)

        self.assertEqual(outputs["voxels"].shape, (0, 2, 4))
        self.assertEqual(outputs["num_points"].shape, (0,))
        self.assertEqual(outputs["voxel_coords"].shape, (0, 4))
        self.assertEqual(outputs["point_voxel_indices"].shape, (0,))
        self.assertEqual(int(outputs["num_dropped_voxels"]), 0)

    def test_eval_mode_uses_eval_max_voxels_budget(self) -> None:
        """
        Test that the voxel budget switches with ``is_training``: training truncates at
        ``max_voxels`` while evaluation keeps pillars up to ``eval_max_voxels``.
        """
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=1,
            eval_max_voxels=8,
            voxelization_z_order_first=True,
        )
        # Three points in three distinct pillars
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],
                [1.1, 1.1, 0.0, 2.0],
                [2.1, 2.1, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )

        train_outputs = preprocessor({"points": [points]}, is_training=True)
        self.assertEqual(train_outputs["voxels"].shape[0], 1)

        eval_outputs = preprocessor({"points": [points]}, is_training=False)
        self.assertEqual(eval_outputs["voxels"].shape[0], 3)

    def test_eval_mode_without_eval_max_voxels_raises(self) -> None:
        """
        Test that running in evaluation mode without an explicit ``eval_max_voxels`` raises
        instead of silently reusing the training budget.
        """
        batch = {"points": [torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)]}

        with self.assertRaises(ValueError):
            self.point_pillar_preprocessor(batch, is_training=False)

    def test_train_mode_does_not_require_eval_max_voxels(self) -> None:
        """
        Test that training-mode forward keeps working when ``eval_max_voxels`` is not set,
        so existing training configs stay valid.
        """
        batch = {"points": [torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)]}

        outputs = self.point_pillar_preprocessor(batch, is_training=True)

        self.assertEqual(outputs["voxels"].shape[0], 1)

    def test_passthrough_of_existing_keys(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly passes through existing
        keys in the input batch dictionary.
        """
        sentinel = torch.tensor([42.0])
        batch = {
            "points": [torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)],
            "gt_boxes": sentinel,
        }
        outputs = self.point_pillar_preprocessor(batch, is_training=True)
        self.assertIs(outputs["gt_boxes"], sentinel)


if __name__ == "__main__":
    unittest.main()
