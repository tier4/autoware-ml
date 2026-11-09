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

"""Unit tests for the shared segmentation eval-output builder."""

import unittest

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.metrics.segmentation3d.eval_output import (
    concat_frame_ids,
    segmentation3d_frames_eval_output,
)
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_detection3d_gt_batch,
)


def _batch_inputs(num_frames: int) -> ModelBatchInputs:
    """Model inputs carrying the frame metadata and the boxes of every frame."""
    batch_inputs = build_batch_inputs(
        detection=build_detection3d_gt_batch(
            [torch.zeros((1, 9)), torch.zeros((0, 9))][:num_frames],
            [torch.tensor([0]), torch.zeros((0,), dtype=torch.long)][:num_frames],
        )
    )
    frame_meta = FrameMetaBatch(
        ego2globals=torch.eye(4).repeat(num_frames, 1, 1),
        scene_tokens=["scene-a", "scene-b"][:num_frames],
        timestamps=torch.zeros(num_frames, dtype=torch.float64),
        lanelet_map_paths=["/maps/a.osm", None][:num_frames],
    )
    gt_batch = batch_inputs.multi_task_gt_batch._replace(frame_meta_batch=frame_meta)
    return batch_inputs.replace(multi_task_gt_batch=gt_batch)


class TestConcatFrameIds(unittest.TestCase):
    """The frame index of every point follows the cumulative point offsets."""

    def test_buckets_points_by_offset(self) -> None:
        """
        Input: two frames with 3 and 2 sampled points, so the offset is [3, 5], and the inverse
        mapping of six original points into the sampled indices.
        Expected: every original point resolves to the frame of the sampled point it maps to.
        Check: the frame ids.
        """
        offset = torch.tensor([3, 5])
        inverse = torch.tensor([0, 2, 2, 3, 4, 4])
        self.assertEqual(concat_frame_ids(offset, inverse).tolist(), [0, 0, 0, 1, 1, 1])


class TestSegmentation3DFramesEvalOutput(unittest.TestCase):
    """Batch-concatenated point tensors become one typed frame per sample."""

    def test_splits_the_points_and_attaches_the_frame_meta_and_boxes(self) -> None:
        """
        Input: four points, two per frame, with the metadata and the boxes of two frames.
        Expected: two frames holding their own points, each with its scene token and map path,
        the first with one box and the second with none.
        Check: the frame count, the labels, the metadata and the box shapes of both frames.
        """
        coord = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        pred = torch.tensor([0, 1, 1, 0])
        target = torch.tensor([0, 1, 0, 0])
        scores = torch.full((4, 2), 0.5)
        frame_ids = torch.tensor([0, 0, 1, 1])

        output = segmentation3d_frames_eval_output(
            coord, pred, target, scores, frame_ids, 2, _batch_inputs(num_frames=2)
        )

        self.assertEqual(len(output.frames), 2)
        first, second = output.frames
        self.assertEqual(first.pred.tolist(), [0, 1])
        self.assertEqual(second.target.tolist(), [0, 0])
        assert first.frame_meta is not None and second.frame_meta is not None
        self.assertEqual(first.frame_meta.scene_token, "scene-a")
        self.assertEqual(second.frame_meta.scene_token, "scene-b")
        self.assertEqual(first.frame_meta.lanelet_map_path, "/maps/a.osm")
        self.assertIsNone(second.frame_meta.lanelet_map_path)
        assert first.gt_boxes is not None and second.gt_box_labels is not None
        self.assertEqual(first.gt_boxes.shape, (1, 9))
        self.assertEqual(second.gt_box_labels.shape, (0,))
        self.assertTrue(output.carries_frame_meta)
        self.assertTrue(output.carries_scores)
        self.assertTrue(output.carries_gt_boxes)

    def test_rejects_frame_meta_that_does_not_align_with_the_frames(self) -> None:
        """
        Input: two frames of points with the metadata of a single frame.
        Expected: the builder refuses the batch, or points would get another frame's context.
        Check: ValueError naming the ego pose entries.
        """
        coord = torch.zeros((2, 3))
        labels = torch.zeros(2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "ego2global"):
            segmentation3d_frames_eval_output(
                coord,
                labels,
                labels,
                torch.full((2, 2), 0.5),
                torch.tensor([0, 1]),
                2,
                _batch_inputs(num_frames=1),
            )


if __name__ == "__main__":
    unittest.main()
