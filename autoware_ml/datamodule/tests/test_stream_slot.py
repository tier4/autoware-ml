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

"""Unit tests for the stream slots a dataset and a split augment a stream with."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import polars as pl

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset import (
    BaseDataset,
    ConcatDataset,
    PaddingSlot,
    StreamPaddingSlot,
    StreamSlot,
)
from autoware_ml.transforms.base import BaseTransform, TransformsCompose


class _Rotate(BaseTransform):
    """Record the angle applied to every sample, kept for its stream or drawn per sample."""

    def __init__(self) -> None:
        super().__init__()
        self.parameters: float | None = None
        # Angles applied and angles drawn, shared with the copies a pipeline keeps per stream
        self.draws: list[float] = []
        self.sampled: list[float] = []

    def sample_parameters(self) -> float:
        self.sampled.append(float(np.random.uniform()))
        return self.sampled[-1]

    def draw_parameters(self) -> bool:
        self.parameters = self.sample_parameters()
        return True

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        parameters = self.parameters if self.parameters is not None else self.sample_parameters()
        return self.transform_with_parameters(model_gt_sample, parameters)

    def transform_with_parameters(
        self, model_gt_sample: ModelGTSample, parameters: float
    ) -> ModelGTSample:
        self.draws.append(parameters)
        return model_gt_sample


class _AugmentedDataset(BaseDataset):
    """Dataset of two empty samples, behind a pipeline of one random transform if asked."""

    def __init__(self, with_pipeline: bool = True) -> None:
        self.rotate = _Rotate()
        super().__init__(
            database_root_path="/",
            dataset_records_dataframe=pl.DataFrame({"index": [0, 1]}),
            transforms=TransformsCompose([self.rotate]) if with_pipeline else None,
        )

    def get_data_sample(self, index: int) -> ModelGTSample:
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def draw(self, index: int) -> float:
        """The draw the pipeline makes for one index."""
        self[index]
        return self.rotate.draws[-1]


class TestStreamSlot(unittest.TestCase):
    """The augmentation a dataset applies to the frames of a stream."""

    STREAM = (0, 0, 0)
    OTHER_STREAM = (0, 0, 1)

    def setUp(self) -> None:
        """Set up the dataset behind one random transform shared by the tests."""
        self.dataset = _AugmentedDataset()

    def test_frames_of_a_stream_are_augmented_alike_and_other_frames_on_their_own(self) -> None:
        """
        Input: a dataset behind one random transform, read with stream slots of two streams
        and then with plain indices.
        Expected: the frames of a stream share one draw, another stream gets another one,
        and a plain index draws for itself again.
        Check: the three frames of one stream record one value, the other stream another
        one, and two plain reads of one frame record different values.
        """
        in_stream = [self.dataset.draw(StreamSlot(index, self.STREAM)) for index in (0, 1, 0)]
        other_stream = self.dataset.draw(StreamSlot(0, self.OTHER_STREAM))
        on_their_own = [self.dataset.draw(0), self.dataset.draw(0)]

        self.assertEqual(len(set(in_stream)), 1)
        self.assertNotEqual(other_stream, in_stream[0])
        self.assertNotEqual(on_their_own[0], on_their_own[1])

    def test_streams_read_in_turn_keep_their_own_augmentation(self) -> None:
        """
        Input: the frames of two streams read in turn, as a dataloader worker reads the
        lanes of a batch.
        Expected: the pipeline draws the augmentation of a stream once and keeps it while
        the other stream is read.
        Check: the two reads of either stream record one value, the streams differ, and
        the transform drew twice in all.
        """
        first = self.dataset.draw(StreamSlot(0, self.STREAM))
        other = self.dataset.draw(StreamSlot(1, self.OTHER_STREAM))
        first_again = self.dataset.draw(StreamSlot(1, self.STREAM))
        other_again = self.dataset.draw(StreamSlot(0, self.OTHER_STREAM))

        self.assertEqual(first_again, first)
        self.assertEqual(other_again, other)
        self.assertNotEqual(first, other)
        self.assertEqual(len(self.dataset.rotate.sampled), 2)

    def test_a_dataset_without_a_pipeline_serves_a_stream_slot(self) -> None:
        """
        Input: a dataset without a transform pipeline, read with a stream slot.
        Expected: there is no augmentation to draw, and the frame is still served.
        Check: the slot is served as a real frame.
        """
        dataset = _AugmentedDataset(with_pipeline=False)

        self.assertFalse(dataset[StreamSlot(1, self.STREAM)].is_padding)

    def test_a_stream_slot_is_served_as_the_frame_or_the_padding_it_stands_for(self) -> None:
        """
        Input: a stream slot and a stream padding slot of the same frame.
        Expected: only the padding slot is served as padding.
        Check: is_padding is False for the stream slot and True for the padding slot.
        """
        self.assertFalse(self.dataset[StreamSlot(1, self.STREAM)].is_padding)
        self.assertTrue(self.dataset[StreamPaddingSlot(1, self.STREAM)].is_padding)

    def test_a_split_hands_its_sources_the_stream_of_a_slot(self) -> None:
        """
        Input: a split of two datasets of two samples, and stream slots of one stream on
        index 0 and 3, one in each source.
        Expected: the split forwards the seed of the slot to the source serving it, so both
        sources draw the augmentation of the stream.
        Check: both sources record one and the same value.
        """
        second = _AugmentedDataset()
        split = ConcatDataset([self.dataset, second], repeats=[1, 1])

        split[StreamSlot(0, self.STREAM)]
        split[StreamSlot(3, self.STREAM)]

        self.assertEqual(len(self.dataset.rotate.draws), 1)
        self.assertEqual(second.rotate.draws, self.dataset.rotate.draws)

    def test_a_split_hands_its_sources_the_padding_of_a_slot(self) -> None:
        """
        Input: a split of two datasets of two samples, read with a stream padding slot and a
        plain padding slot on index 3, the second sample of the second source.
        Expected: the source is handed a slot of the same kind on its own index.
        Check: the source receives a StreamPaddingSlot on index 1 carrying the seed of the
        stream, then a plain PaddingSlot.
        """
        split = ConcatDataset([self.dataset, _AugmentedDataset()], repeats=[1, 1])

        served: list[int] = []
        with mock.patch.object(
            _AugmentedDataset, "__getitem__", lambda self, index: served.append(index)
        ):
            split[StreamPaddingSlot(3, self.STREAM)]
            split[PaddingSlot(3)]

        self.assertIsInstance(served[0], StreamPaddingSlot)
        self.assertEqual(served[0].stream_seed, self.STREAM)
        self.assertEqual(served[0], 1)
        self.assertIs(type(served[1]), PaddingSlot)


if __name__ == "__main__":
    unittest.main()
