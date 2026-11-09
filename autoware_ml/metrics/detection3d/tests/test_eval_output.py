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

"""Unit tests for the detection eval-output builder."""

import unittest

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output


class TestTypedDetection3DEvalOutput(unittest.TestCase):
    """Unit tests for the detection3d_eval_output function."""

    def setUp(self) -> None:
        """Set up the common inputs for the tests."""
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
                    labels_3d=torch.tensor([1], dtype=torch.int64, device=self.device),
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
        self.detection3d_gt_batch = Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            gt_bboxes_num_points=gt_bboxes_num_points,
        )
        self.multi_task_batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=self.detection3d_gt_batch,
                segmentation3d_gt_batch=None,
                image_gt_batch=None,
                padding_mask=torch.zeros(2, dtype=torch.bool, device=self.device),
            ),
            voxels_data=None,
            image_data=None,
            range_view_data=None,
        )

    def _with_gt_batch(self, gt_batch: ModelGTBatch) -> ModelBatchInputs:
        """The common inputs with another ground truth batch."""
        return ModelBatchInputs(
            multi_task_gt_batch=gt_batch, voxels_data=None, image_data=None, range_view_data=None
        )

    def test_rejects_a_batch_without_detection_ground_truth(self) -> None:
        """
        Input: a batch whose ground truth carries no detection part.
        Expected: the builder refuses it, there is nothing to pair the predictions with.
        Check: ValueError.
        """
        batch_inputs = self._with_gt_batch(
            ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=None,
                segmentation3d_gt_batch=None,
                image_gt_batch=None,
                padding_mask=torch.zeros(1, dtype=torch.bool),
            )
        )
        with self.assertRaises(ValueError):
            detection3d_eval_output(
                batch_inputs=batch_inputs, predictions=self.multi_task_predictions
            )

    def test_rejects_predictions_without_a_detection_part(self) -> None:
        """
        Input: model predictions that carry no detection part.
        Expected: the builder refuses them.
        Check: ValueError.
        """
        with self.assertRaises(ValueError):
            detection3d_eval_output(
                batch_inputs=self.multi_task_batch_inputs,
                predictions=ModelPredictions(detection3d_predictions=None),
            )

    def test_pairs_every_sample_with_its_valid_ground_truth(self) -> None:
        """
        Input: two samples with two and one valid ground truth boxes.
        Expected: one frame per sample, the ground truth trimmed to the valid boxes and the
        predictions carried through untouched.
        Check: the frame count, the ground truth tensors and the predictions of every frame.
        """
        output = detection3d_eval_output(
            batch_inputs=self.multi_task_batch_inputs, predictions=self.multi_task_predictions
        )

        gt_batch = self.detection3d_gt_batch
        self.assertEqual(len(output.frames), 2)
        assert self.multi_task_predictions.detection3d_predictions is not None
        for index, frame in enumerate(output.frames):
            num_valid = int(gt_batch.gt_valid_bboxes[index])
            self.assertEqual(frame.gt_boxes.shape[0], num_valid)
            self.assertTrue(torch.equal(frame.gt_boxes, gt_batch.gt_bboxes_3d[index, :num_valid]))
            self.assertTrue(
                torch.equal(frame.gt_labels, gt_batch.gt_labels_3d[index, :num_valid].long())
            )
            assert frame.gt_num_points is not None
            self.assertTrue(
                torch.equal(frame.gt_num_points, gt_batch.gt_bboxes_num_points[index, :num_valid])
            )
            self.assertIs(
                frame.predictions, self.multi_task_predictions.detection3d_predictions[index]
            )
        self.assertTrue(output.carries_gt_num_points)
        self.assertFalse(output.carries_frame_meta)

    def test_frame_meta_is_attached_per_frame(self) -> None:
        """
        Input: the two-sample batch with frame metadata, the second frame without a map.
        Expected: every frame carries its own pose, scene token and map path.
        Check: the metadata of both frames.
        """
        ego2globals = torch.eye(4, device=self.device).repeat(2, 1, 1)
        ego2globals[1, 0, 3] = 7.0
        gt_batch = self.multi_task_batch_inputs.multi_task_gt_batch._replace(
            frame_meta_batch=FrameMetaBatch(
                ego2globals=ego2globals,
                scene_tokens=["db/scene_a/0", "db/scene_b/1"],
                timestamps=torch.tensor([0.0, 0.1], dtype=torch.float64, device=self.device),
                lanelet_map_paths=["/maps/scene_a.osm", None],
            )
        )

        output = detection3d_eval_output(
            batch_inputs=self._with_gt_batch(gt_batch), predictions=self.multi_task_predictions
        )

        self.assertTrue(output.carries_frame_meta)
        first, second = output.frames
        assert first.frame_meta is not None and second.frame_meta is not None
        self.assertTrue(torch.equal(second.frame_meta.ego2global, ego2globals[1]))
        self.assertEqual(first.frame_meta.scene_token, "db/scene_a/0")
        self.assertEqual(second.frame_meta.scene_token, "db/scene_b/1")
        self.assertEqual(first.frame_meta.lanelet_map_path, "/maps/scene_a.osm")
        self.assertIsNone(second.frame_meta.lanelet_map_path)

    def test_flags_the_padding_samples(self) -> None:
        """
        Input: the two-sample batch with its second sample flagged as padding.
        Expected: the padding sample keeps its frame, which every GPU has to hold for the
        gather, and the frame is flagged for the metric to leave out.
        Check: the frame count and the per-frame padding flags.
        """
        gt_batch = self.multi_task_batch_inputs.multi_task_gt_batch._replace(
            padding_mask=torch.tensor([False, True], device=self.device)
        )

        output = detection3d_eval_output(
            batch_inputs=self._with_gt_batch(gt_batch), predictions=self.multi_task_predictions
        )

        self.assertEqual(len(output.frames), 2)
        self.assertEqual([frame.is_padding for frame in output.frames], [False, True])


if __name__ == "__main__":
    unittest.main()
