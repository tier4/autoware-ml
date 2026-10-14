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

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.detection3d.point_pillar_preprocessor import PointPillarPreprocessor


def _inputs(samples: list[torch.Tensor]) -> ModelBatchInputs:
    """Collate the points of every sample into model inputs."""
    batch_indices = torch.cat(
        [
            torch.full((sample.shape[0],), index, dtype=torch.int32)
            for index, sample in enumerate(samples)
        ]
    )
    points = PointCloudGTBatch(
        points=torch.cat(samples),
        batch_indices=batch_indices,
        batch_size=len(samples),
        timestamp_difference_dim=-1,
    )
    return ModelBatchInputs.from_gt_batch(
        ModelGTBatch(
            point_cloud_gt_batch=points,
            detection3d_gt_batch=None,
            segmentation3d_gt_batch=None,
            image_gt_batch=None,
        )
    )


class TestPointPillarPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PointPillarPreprocessor instance for all tests. Note that this class will
        be called in each test case.
        """
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.point_pillar_preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=8,
            eval_max_voxels=8,
        )

    def test_forward_builds_padded_pillars(self) -> None:
        """
        Test that the forward method correctly builds padded pillars from a batch of point
        clouds.
        """
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],
                [0.2, 0.2, 0.0, 2.0],
                [1.1, 1.1, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )

        voxels = self.point_pillar_preprocessor(_inputs([points]), is_training=True).voxels_data

        assert voxels is not None
        self.assertEqual(voxels.voxels.shape, (2, 2, 4))
        self.assertEqual(voxels.num_points.tolist(), [2, 1])
        self.assertEqual(voxels.coords.tolist(), [[0, 0, 0], [1, 1, 0]])
        self.assertEqual(voxels.batch_indices.tolist(), [0, 0])

    def test_batch_index_increments_per_sample(self) -> None:
        """
        Test that the sample index of the voxels increments for each sample in the batch.
        """
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)

        voxels = self.point_pillar_preprocessor(
            _inputs([point, point, point]), is_training=True
        ).voxels_data

        assert voxels is not None
        self.assertEqual(voxels.batch_indices.tolist(), [0, 1, 2])

    def test_empty_sample_in_batch(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly handles a batch containing an empty
        sample.
        """
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        empty = torch.zeros((0, 4), dtype=torch.float32)

        voxels = self.point_pillar_preprocessor(
            _inputs([point, empty, point]), is_training=True
        ).voxels_data

        # Two non-empty samples  2 voxels total
        assert voxels is not None
        self.assertEqual(voxels.voxels.shape[0], 2)
        self.assertEqual(set(voxels.batch_indices.tolist()), {0, 2})

    def test_batch_without_points_returns_empty_pillars(self) -> None:
        """
        Test that the PointPillarPreprocessor returns empty pillars when the batch holds no
        points.
        """
        empty = torch.zeros((0, 4), dtype=torch.float32)

        voxels = self.point_pillar_preprocessor(_inputs([empty]), is_training=True).voxels_data

        assert voxels is not None
        self.assertEqual(voxels.voxels.shape, (0, 2, 4))
        self.assertEqual(voxels.num_points.shape, (0,))
        self.assertEqual(voxels.coords.shape, (0, 3))

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

        train_voxels = preprocessor(_inputs([points]), is_training=True).voxels_data
        eval_voxels = preprocessor(_inputs([points]), is_training=False).voxels_data

        assert train_voxels is not None and eval_voxels is not None
        self.assertEqual(train_voxels.voxels.shape[0], 1)
        self.assertEqual(eval_voxels.voxels.shape[0], 3)

    def test_requires_an_explicit_eval_max_voxels(self) -> None:
        """
        Test that the evaluation voxel budget must be stated when the preprocessor is built,
        instead of silently reusing the training budget at inference.
        """
        with self.assertRaises(TypeError):
            PointPillarPreprocessor(  # type: ignore[call-arg]
                voxel_size=[1.0, 1.0, 4.0],
                point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
                max_num_points=2,
                max_voxels=8,
            )

    def test_batch_without_point_cloud_raises(self) -> None:
        """
        Test that a batch carrying no point cloud is rejected.
        """
        inputs = ModelBatchInputs.from_gt_batch(
            ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=None,
                segmentation3d_gt_batch=None,
                image_gt_batch=None,
            )
        )

        with self.assertRaises(ValueError):
            self.point_pillar_preprocessor(inputs, is_training=True)

    def test_keeps_the_rest_of_the_inputs(self) -> None:
        """
        Test that the PointPillarPreprocessor only adds the voxels to the model inputs.
        """
        inputs = _inputs([torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)])

        outputs = self.point_pillar_preprocessor(inputs, is_training=True)

        self.assertIs(outputs.multi_task_gt_batch, inputs.multi_task_gt_batch)
        self.assertIsNone(outputs.image_data)


if __name__ == "__main__":
    unittest.main()
