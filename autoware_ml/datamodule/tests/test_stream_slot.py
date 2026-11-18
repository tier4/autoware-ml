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

from dataclasses import dataclass
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
from autoware_ml.transforms.base import BaseTransform, TransformParameters, TransformsCompose


@dataclass(frozen=True)
class _Angle(TransformParameters):
    """The angle drawn by ``_Rotate``."""

    angle: float


class _Rotate(BaseTransform[_Angle]):
    """Record the angle applied to every sample, drawn per sample or handed in by its stream."""

    def __init__(self) -> None:
        super().__init__()
        self.draws: list[float] = []

    def sample_parameters(self) -> _Angle:
        return _Angle(angle=float(np.random.uniform()))

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        return self.transform_with_parameters(model_gt_sample, self.sample_parameters())

    def transform_with_parameters(
        self, model_gt_sample: ModelGTSample, parameters: _Angle
    ) -> ModelGTSample:
        self.draws.append(parameters.angle)
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

    def setUp(self) -> None:
        """Set up the dataset behind one random transform shared by the tests."""
        self.dataset = _AugmentedDataset()

    def test_frames_of_a_stream_are_augmented_alike_and_other_frames_on_their_own(self) -> None:
        """
        Input: a dataset behind one random transform, two streams drawn from it, and frames
        served as stream slots of either stream or as plain indices.
        Expected: a stream slot is augmented by the draw of its stream, a plain index draws
        for itself.
        Check: the three frames of one stream record its draw, the other stream records
        another one, and two plain reads of one frame record different draws.
        """
        stream = self.dataset.sample_stream_augmentation(0)
        other = self.dataset.sample_stream_augmentation(0)

        in_stream = [self.dataset.draw(StreamSlot(index, stream)) for index in (0, 1, 0)]
        other_stream = self.dataset.draw(StreamSlot(0, other))
        on_their_own = [self.dataset.draw(0), self.dataset.draw(0)]

        self.assertEqual(set(in_stream), {stream[0].values.angle})
        self.assertEqual(other_stream, other[0].values.angle)
        self.assertNotEqual(other_stream, in_stream[0])
        self.assertNotEqual(on_their_own[0], on_their_own[1])

    def test_a_dataset_without_a_pipeline_draws_no_stream_augmentation(self) -> None:
        """
        Input: a dataset without a transform pipeline.
        Expected: there is nothing to draw for a stream, and a stream slot is still served.
        Check: the stream augmentation is empty and the slot is served as a real frame.
        """
        dataset = _AugmentedDataset(with_pipeline=False)

        self.assertEqual(dataset.sample_stream_augmentation(0), ())
        self.assertFalse(dataset[StreamSlot(1, ())].is_padding)

    def test_a_stream_slot_is_served_as_the_frame_or_the_padding_it_stands_for(self) -> None:
        """
        Input: a stream slot and a stream padding slot of the same frame.
        Expected: only the padding slot is served as padding.
        Check: is_padding is False for the stream slot and True for the padding slot.
        """
        stream = self.dataset.sample_stream_augmentation(0)

        self.assertFalse(self.dataset[StreamSlot(1, stream)].is_padding)
        self.assertTrue(self.dataset[StreamPaddingSlot(1, stream)].is_padding)

    def test_a_split_hands_its_sources_the_augmentation_of_a_slot(self) -> None:
        """
        Input: a split of two datasets of two samples, and stream slots on index 0 and 3,
        one in each source.
        Expected: the split forwards the augmentation of the slot to the source serving it.
        Check: both sources record the one draw of the stream.
        """
        second = _AugmentedDataset()
        split = ConcatDataset([self.dataset, second], repeats=[1, 1])
        stream = split.sample_stream_augmentation(3)

        split[StreamSlot(0, stream)]
        split[StreamSlot(3, stream)]

        self.assertEqual(self.dataset.rotate.draws, [stream[0].values.angle])
        self.assertEqual(second.rotate.draws, [stream[0].values.angle])

    def test_a_split_hands_its_sources_the_padding_of_a_slot(self) -> None:
        """
        Input: a split of two datasets of two samples, read with a stream padding slot and a
        plain padding slot on index 3, the second sample of the second source.
        Expected: the source is handed a slot of the same kind on its own index.
        Check: the source receives a StreamPaddingSlot on index 1 carrying the augmentation,
        then a plain PaddingSlot.
        """
        split = ConcatDataset([self.dataset, _AugmentedDataset()], repeats=[1, 1])
        stream = split.sample_stream_augmentation(3)

        served: list[int] = []
        with mock.patch.object(
            _AugmentedDataset, "__getitem__", lambda self, index: served.append(index)
        ):
            split[StreamPaddingSlot(3, stream)]
            split[PaddingSlot(3)]

        self.assertIsInstance(served[0], StreamPaddingSlot)
        self.assertEqual(served[0].augmentation, stream)
        self.assertEqual(served[0], 1)
        self.assertIs(type(served[1]), PaddingSlot)


if __name__ == "__main__":
    unittest.main()
