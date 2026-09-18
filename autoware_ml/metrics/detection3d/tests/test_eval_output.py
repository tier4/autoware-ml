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

"""Unit tests for eval_output."""

import unittest

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.batch.detection3d import (
    Detection3DGTBatch,
)
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.metrics.detection3d.eval_output import multi_task_eval_output


class TestMultiTaskEvalOutput(unittest.TestCase):
    """Unit tests for the multi_task_eval_output function."""

    def setUp(self) -> None:
        """Set up the common inputs for the tests."""
        # Create dummy ModelPredictions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.multi_task_predictions = ModelPredictions(
            detection3d_predictions=[
                Detection3DSamplePredictions(
                    bboxes_3d=torch.tensor(
                        [
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    scores_3d=torch.tensor([0.9, 0.0], dtype=torch.float32, device=self.device),
                    labels_3d=torch.tensor([1, 0], dtype=torch.int64, device=self.device),
                ),
                Detection3DSamplePredictions(
                    bboxes_3d=torch.tensor(
                        [[4.9, 0.2, 0.1, 6.8, 7.2, 9.2, 40.0, 50.0, 60.0]], device=self.device
                    ),
                    scores_3d=torch.tensor([0.9], dtype=torch.float32, device=self.device),
                    labels_3d=torch.tensor([1, 2], dtype=torch.int64, device=self.device),
                ),
            ]
        )
        # (batch_size, num_boxes, box_dim) = (2, 2, 10)
        gt_bboxes_3d = torch.tensor(
            [
                [
                    [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 5.0, 6.0, 7.0, 10.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 30.0, 40.0, 50.0],
                ],
                [
                    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 30.0, 40.0, 50.0, 60.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        gt_labels_3d = torch.tensor([[1, 2], [3, -1]], dtype=torch.int32, device=self.device)
        gt_valid_bboxes = torch.tensor([2, 1], dtype=torch.int32, device=self.device)
        gt_bboxes_num_points = torch.tensor(
            [[100, 200], [300, 0]], dtype=torch.int32, device=self.device
        )

        # Inputs
        detection3d_gt_batch = Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            gt_bboxes_num_points=gt_bboxes_num_points,
        )
        self.multi_task_batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=detection3d_gt_batch,
                image_gt_batch=None,
            ),
            voxels_data=None,
            image_data=None,
        )

    def test_detection3d_gt_batch_assertion(self):
        """Test that multi_task_eval_output raises ValueError when detection3d_gt_batch is None."""

        multi_task_batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=None,
                image_gt_batch=None,
            ),
            voxels_data=None,
            image_data=None,
        )
        with self.assertRaises(ValueError):
            multi_task_eval_output(
                multi_task_batch_inputs=multi_task_batch_inputs,
                multi_task_predictions=self.multi_task_predictions,
            )

    def test_detection3d_predictions_assertion(self):
        """Test that multi_task_eval_output raises ValueError when detection3d_predictions is None."""

        multi_task_predictions = ModelPredictions(detection3d_predictions=None)
        with self.assertRaises(ValueError):
            multi_task_eval_output(
                multi_task_batch_inputs=self.multi_task_batch_inputs,
                multi_task_predictions=multi_task_predictions,
            )

    def test_eval_outputs(self):
        """Test that multi_task_eval_output correctly pairs predictions with ground truth."""

        eval_outputs = multi_task_eval_output(
            multi_task_batch_inputs=self.multi_task_batch_inputs,
            multi_task_predictions=self.multi_task_predictions,
        )

        self.assertIn("predictions", eval_outputs)
        self.assertIn("gt_boxes", eval_outputs)
        self.assertIn("gt_labels", eval_outputs)
        self.assertIn("gt_num_points", eval_outputs)

        # Ground truth comes back as one tensor per sample, trimmed to the valid boxes, which is
        # the layout the detection metric consumes.
        detection3d_gt_batch = self.multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        assert detection3d_gt_batch is not None
        batch_size = detection3d_gt_batch.gt_valid_bboxes.shape[0]
        self.assertEqual(len(eval_outputs["gt_boxes"]), batch_size)
        self.assertEqual(len(eval_outputs["gt_labels"]), batch_size)
        self.assertEqual(len(eval_outputs["gt_num_points"]), batch_size)
        for batch_idx in range(batch_size):
            num_valid = int(detection3d_gt_batch.gt_valid_bboxes[batch_idx])
            self.assertEqual(eval_outputs["gt_boxes"][batch_idx].shape[0], num_valid)
            self.assertTrue(
                torch.equal(
                    eval_outputs["gt_boxes"][batch_idx],
                    detection3d_gt_batch.gt_bboxes_3d[batch_idx, :num_valid],
                )
            )
            self.assertTrue(
                torch.equal(
                    eval_outputs["gt_labels"][batch_idx],
                    detection3d_gt_batch.gt_labels_3d[batch_idx, :num_valid],
                )
            )
            self.assertTrue(
                torch.equal(
                    eval_outputs["gt_num_points"][batch_idx],
                    detection3d_gt_batch.gt_bboxes_num_points[batch_idx, :num_valid],
                )
            )

        assert self.multi_task_predictions.detection3d_predictions is not None
        for batch_idx in range(len(eval_outputs["predictions"])):
            self.assertTrue(
                torch.allclose(
                    eval_outputs["predictions"][batch_idx]["bboxes_3d"],
                    self.multi_task_predictions.detection3d_predictions[batch_idx].bboxes_3d,
                )
            )
            self.assertTrue(
                torch.allclose(
                    eval_outputs["predictions"][batch_idx]["scores_3d"],
                    self.multi_task_predictions.detection3d_predictions[batch_idx].scores_3d,
                )
            )
            self.assertTrue(
                torch.allclose(
                    eval_outputs["predictions"][batch_idx]["labels_3d"],
                    self.multi_task_predictions.detection3d_predictions[batch_idx].labels_3d,
                )
            )

    def test_frame_meta_is_passed_through_per_frame(self):
        """Test that ego2global and scene_token are handed to the metric per frame."""
        ego2globals = torch.eye(4, device=self.device).repeat(2, 1, 1)
        ego2globals[1, 0, 3] = 7.0
        gt_batch = self.multi_task_batch_inputs.multi_task_gt_batch._replace(
            frame_meta_batch=FrameMetaBatch(
                ego2globals=ego2globals, scene_tokens=["db/scene_a/0", "db/scene_b/1"]
            )
        )
        multi_task_batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=gt_batch, voxels_data=None, image_data=None
        )

        eval_outputs = multi_task_eval_output(
            multi_task_batch_inputs=multi_task_batch_inputs,
            multi_task_predictions=self.multi_task_predictions,
        )

        self.assertEqual(len(eval_outputs["ego2global"]), 2)
        self.assertTrue(torch.equal(eval_outputs["ego2global"][1], ego2globals[1]))
        self.assertEqual(eval_outputs["scene_token"], ["db/scene_a/0", "db/scene_b/1"])

    def test_frame_meta_keys_absent_without_frame_meta_batch(self):
        """Test that the metadata keys are left out when the batch carries none."""
        eval_outputs = multi_task_eval_output(
            multi_task_batch_inputs=self.multi_task_batch_inputs,
            multi_task_predictions=self.multi_task_predictions,
        )

        self.assertNotIn("ego2global", eval_outputs)
        self.assertNotIn("scene_token", eval_outputs)


if __name__ == "__main__":
    unittest.main()
