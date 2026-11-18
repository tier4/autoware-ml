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

"""Unit tests for the 2D box padding size a dataset and a split collate with."""

from __future__ import annotations

from collections.abc import Sequence
import unittest
from unittest import mock

import numpy as np
import polars as pl
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset import (
    BaseDataset,
    ConcatDataset,
    PaddingSlot,
    StreamPaddingSlot,
    StreamSlot,
)
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.transforms.base import BaseTransform, TransformsCompose
from autoware_ml.types.geometry import Box2DFieldIndex


class _Dataset(BaseDataset):
    """Dataset serving samples of two cameras holding ``counts`` 2D boxes."""

    def __init__(self, max_num_2d_gt_bboxes: int | None, counts: Sequence[int] = (1, 0)) -> None:
        super().__init__(
            database_root_path="/",
            max_num_2d_gt_bboxes=max_num_2d_gt_bboxes,
            dataset_records_dataframe=pl.DataFrame({"index": [0, 1]}),
            transforms=None,
        )
        self.counts = counts

    def get_data_sample(self, index: int) -> ModelGTSample:
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
            detection2d_gt_sample=Detection2DGTSample(
                gt_bboxes_2d=[torch.ones((count, len(Box2DFieldIndex))) for count in self.counts],
                gt_labels_2d=[torch.zeros(count, dtype=torch.int32) for count in self.counts],
            ),
        )


class TestMaxNum2DGtBboxes(unittest.TestCase):
    """The number of 2D boxes every camera is padded to in a batch."""

    def test_a_dataset_pads_every_camera_to_its_max_num_2d_gt_bboxes(self) -> None:
        """
        Input: a dataset setting max_num_2d_gt_bboxes 4, serving two cameras holding 1 and 0
        boxes.
        Expected: collate_fn pads every camera to the 4 boxes the dataset sets.
        Check: the collated boxes have 4 rows per camera and the valid counts are 1 and 0.
        """
        dataset = _Dataset(max_num_2d_gt_bboxes=4)

        batch = dataset.collate_fn([dataset[0], dataset[1]]).detection2d_gt_batch

        assert batch is not None
        self.assertEqual(tuple(batch.gt_bboxes_2d.shape), (2, 2, 4, len(Box2DFieldIndex)))
        self.assertEqual(batch.gt_valid_bboxes_2d.tolist(), [[1, 0], [1, 0]])

    def test_a_split_pads_to_the_largest_size_of_its_sources(self) -> None:
        """
        Input: a split of three sources setting max_num_2d_gt_bboxes 3, none and 5.
        Expected: a batch mixes the sources, so the split pads to the largest size any of
        them sets, as it does for the 3D boxes.
        Check: the split keeps 5 and its collated boxes have 5 rows per camera.
        """
        split = ConcatDataset(
            [_Dataset(3), _Dataset(None), _Dataset(5, counts=(2, 1))], repeats=[1, 1, 1]
        )

        batch = split.collate_fn([split[0], split[5]]).detection2d_gt_batch

        self.assertEqual(split.max_num_2d_gt_bboxes, 5)
        assert batch is not None
        self.assertEqual(tuple(batch.gt_labels_2d.shape), (2, 2, 5))
        self.assertEqual(batch.gt_valid_bboxes_2d.tolist(), [[1, 0], [2, 1]])

    def test_a_split_without_a_size_cannot_collate_2d_boxes(self) -> None:
        """
        Input: a split whose only source sets no max_num_2d_gt_bboxes but serves 2D boxes.
        Expected: the batch has no size to pad the 2D boxes to, which is a configuration error
        rather than a reason to fall back to the largest camera of the batch.
        Check: collate_fn raises a ValueError naming max_num_2d_gt_bboxes.
        """
        split = ConcatDataset([_Dataset(None)], repeats=[1])

        self.assertIsNone(split.max_num_2d_gt_bboxes)
        with self.assertRaisesRegex(ValueError, "max_num_2d_gt_bboxes"):
            split.collate_fn([split[0]])

    def test_the_t4_dataset_takes_the_size_from_its_config(self) -> None:
        """
        Input: a T4Dataset declared with max_num_2d_gt_bboxes 7, as a dataset config sets it.
        Expected: the T4 dataset hands the size to the base dataset its collation reads.
        Check: the dataset keeps 7.
        """
        dataset = T4Dataset(
            database_root_path="/",
            max_num_2d_gt_bboxes=7,
            transforms=None,
            dataset_tasks={},
            lidar_intensity_scale=255.0,
        )

        self.assertEqual(dataset.max_num_2d_gt_bboxes, 7)


if __name__ == "__main__":
    unittest.main()


class _Rotate(BaseTransform):
    """Record the angle applied to every sample, drawn per sample or handed in by its stream."""

    def __init__(self) -> None:
        super().__init__()
        self.draws: list[float] = []

    def sample_parameters(self) -> float:
        return float(np.random.uniform())

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        return self.transform_with_parameters(model_gt_sample, self.sample_parameters())

    def transform_with_parameters(
        self, model_gt_sample: ModelGTSample, parameters: float
    ) -> ModelGTSample:
        self.draws.append(parameters)
        return model_gt_sample


class _AugmentedDataset(_Dataset):
    """The two record dataset behind a pipeline of one random transform."""

    def __init__(self) -> None:
        super().__init__(max_num_2d_gt_bboxes=1)
        self.rotate = _Rotate()
        self.transforms = TransformsCompose([self.rotate])

    def draw(self, index: int) -> float:
        """The draw the pipeline makes for one index."""
        self[index]
        return self.rotate.draws[-1]


def test_frames_of_a_stream_are_augmented_alike_and_other_frames_on_their_own() -> None:
    dataset = _AugmentedDataset()
    stream, other = dataset.sample_stream_augmentation(0), dataset.sample_stream_augmentation(0)

    in_stream = [dataset.draw(StreamSlot(index, stream)) for index in (0, 1, 0)]
    other_stream = dataset.draw(StreamSlot(0, other))
    on_their_own = [dataset.draw(0), dataset.draw(0)]

    assert set(in_stream) == {stream[0].values}
    assert other_stream == other[0].values != in_stream[0]
    assert on_their_own[0] != on_their_own[1]


def test_a_dataset_without_a_pipeline_draws_no_stream_augmentation() -> None:
    dataset = _Dataset(max_num_2d_gt_bboxes=1)

    assert dataset.sample_stream_augmentation(0) == ()
    assert dataset[StreamSlot(1, ())].is_padding is False


def test_a_stream_slot_is_served_as_the_frame_or_the_padding_it_stands_for() -> None:
    dataset = _AugmentedDataset()
    stream = dataset.sample_stream_augmentation(0)

    assert dataset[StreamSlot(1, stream)].is_padding is False
    assert dataset[StreamPaddingSlot(1, stream)].is_padding is True


def test_a_concatenated_split_hands_its_sources_the_augmentation_and_the_padding_of_a_slot() -> (
    None
):
    first, second = _AugmentedDataset(), _AugmentedDataset()
    split = ConcatDataset([first, second], repeats=[1, 1])
    stream = split.sample_stream_augmentation(3)

    split[StreamSlot(0, stream)]
    split[StreamSlot(3, stream)]
    assert first.rotate.draws == second.rotate.draws == [stream[0].values]

    served: list[int] = []
    with mock.patch.object(_Dataset, "__getitem__", lambda self, index: served.append(index)):
        split[StreamPaddingSlot(3, stream)]
        split[PaddingSlot(3)]
    assert isinstance(served[0], StreamPaddingSlot) and served[0].augmentation == stream
    assert served[0] == 1
    assert type(served[1]) is PaddingSlot
