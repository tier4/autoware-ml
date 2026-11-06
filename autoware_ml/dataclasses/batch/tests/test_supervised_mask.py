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

"""Unit tests for the samples of a batch a model learns from."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch


def _batch(padding: list[bool], key_frames: list[bool] | None) -> ModelGTBatch:
    """Batch with its padding slots and, when given, the key frame flags of its frames."""
    frame_meta = (
        None
        if key_frames is None
        else FrameMetaBatch(
            ego2globals=torch.eye(4).repeat(len(padding), 1, 1),
            scene_tokens=["scene"] * len(padding),
            timestamps=torch.zeros(len(padding), dtype=torch.float64),
            lanelet_map_paths=[None] * len(padding),
            key_frames=key_frames,
        )
    )
    return ModelGTBatch(
        point_cloud_gt_batch=None,
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=None,
        image_gt_batch=None,
        padding_mask=torch.tensor(padding),
        frame_meta_batch=frame_meta,
    )


class TestSupervisedMask(unittest.TestCase):
    def test_a_frame_between_key_frames_takes_no_part(self) -> None:
        """
        Input: a key frame, a frame between key frames and a padding slot.
        Expected: only the key frame is supervised, and only it counts in the logged batch size.
        Check: the mask and the count.
        """
        batch = _batch([False, False, True], [True, False, True])

        self.assertEqual(batch.supervised_mask().tolist(), [True, False, False])
        self.assertEqual(batch.num_supervised_samples(), 1)

    def test_without_frame_metadata_every_real_sample_is_supervised(self) -> None:
        """
        Input: a batch without frame metadata and a padding slot.
        Expected: every sample but the padding slot is supervised.
        Check: the mask.
        """
        self.assertEqual(_batch([False, True], None).supervised_mask().tolist(), [True, False])
