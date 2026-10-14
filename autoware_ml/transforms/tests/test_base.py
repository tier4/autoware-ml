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

"""Tests for the ModelGTSample transform base classes and pipeline context."""

from __future__ import annotations

import unittest

import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.base import (
    BaseTransform,
    ContextAwareTransform,
    PipelineContext,
    TransformsCompose,
)


def _create_sample(tag: float) -> ModelGTSample:
    """Build an empty sample tagged through ``io_processing_time``."""
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
        io_processing_time=tag,
    )


class _FakeDataset:
    """Minimal ``SampleSource`` whose sample ``i`` carries the tag ``i``.

    Every ``apply_transforms`` call is recorded as ``(context.index, transforms)`` in
    ``apply_calls``.
    """

    def __init__(self, length: int) -> None:
        self.length = length
        self.apply_calls: list[tuple[int, TransformsCompose | None]] = []

    def __len__(self) -> int:
        return self.length

    def get_data_sample(self, index: int) -> ModelGTSample:
        return _create_sample(float(index))

    def apply_transforms(
        self,
        multi_task_gt_sample: ModelGTSample,
        transforms: TransformsCompose | None,
        context: PipelineContext,
    ) -> ModelGTSample:
        self.apply_calls.append((context.index, transforms))
        if transforms is None:
            return multi_task_gt_sample
        return transforms(multi_task_gt_sample, context=context)


class _AddTenTransform(BaseTransform):
    """Plain transform: add 10 to the sample tag."""

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        return multi_task_gt_sample._replace(
            io_processing_time=multi_task_gt_sample.io_processing_time + 10.0
        )


class _RecordContextTransform(BaseTransform):
    """Plain transform that records each context it is called with in ``seen``."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[PipelineContext | None] = []

    def __call__(
        self,
        multi_task_gt_sample: ModelGTSample,
        context: PipelineContext | None = None,
    ) -> ModelGTSample:
        self.seen.append(context)
        return super().__call__(multi_task_gt_sample, context=context)

    def transform(self, multi_task_gt_sample: ModelGTSample) -> ModelGTSample:
        return multi_task_gt_sample


class _MixTagsTransform(ContextAwareTransform):
    """Context-aware transform: add the secondary sample's tag to the current one.

    The tag of every secondary sample drawn is recorded in ``secondary_tags``.
    """

    def __init__(self, pre_transform: TransformsCompose | None = None) -> None:
        super().__init__(pre_transform=pre_transform)
        self.secondary_tags: list[float] = []

    def transform(
        self,
        multi_task_gt_sample: ModelGTSample,
        context: PipelineContext,
    ) -> ModelGTSample:
        secondary = self.sample_secondary(context)
        self.secondary_tags.append(secondary.io_processing_time)
        return multi_task_gt_sample._replace(
            io_processing_time=multi_task_gt_sample.io_processing_time
            + secondary.io_processing_time
        )


class _CaptureRngTransform(ContextAwareTransform):
    """Context-aware transform that records each context's generator in ``seen``."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[np.random.Generator] = []

    def transform(
        self,
        multi_task_gt_sample: ModelGTSample,
        context: PipelineContext,
    ) -> ModelGTSample:
        self.seen.append(context.rng)
        return multi_task_gt_sample


class TestTransformsCompose(unittest.TestCase):
    """Context forwarding through the composed pipeline."""

    def test_forwards_context_to_every_transform(self) -> None:
        recorder = _RecordContextTransform()
        context = PipelineContext(dataset=_FakeDataset(3), index=0)

        TransformsCompose([recorder, recorder])(_create_sample(0.0), context=context)

        self.assertEqual(recorder.seen, [context, context])

    def test_plain_transforms_run_without_context(self) -> None:
        pipeline = TransformsCompose([_AddTenTransform()])

        output = pipeline(_create_sample(1.0))

        self.assertEqual(output.io_processing_time, 11.0)


class TestContextAwareTransform(unittest.TestCase):
    """Behavior of transforms that require the pipeline context."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_requires_context(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires a PipelineContext"):
            _MixTagsTransform()(_create_sample(0.0))

    def test_mixes_secondary_sample_through_pipeline(self) -> None:
        dataset = _FakeDataset(2)
        mixer = _MixTagsTransform(pre_transform=TransformsCompose([_AddTenTransform()]))
        pipeline = TransformsCompose([_AddTenTransform(), mixer])
        context = PipelineContext(dataset=dataset, index=0, rng=self.rng)

        mixed = pipeline(dataset.get_data_sample(0), context=context)

        # Primary: 0 + 10. Secondary: sample 1 through the pre-transform, 1 + 10.
        self.assertEqual(mixer.secondary_tags, [11.0])
        self.assertEqual(mixed.io_processing_time, 21.0)


class TestPipelineContext(unittest.TestCase):
    """Secondary sampling and pre-transform materialization."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_sample_secondary_never_returns_current_index(self) -> None:
        context = PipelineContext(dataset=_FakeDataset(4), index=2, rng=self.rng)

        tags = {context.sample_secondary().io_processing_time for _ in range(50)}

        self.assertNotIn(2.0, tags)
        self.assertEqual(tags, {0.0, 1.0, 3.0})

    def test_sample_secondary_reuses_current_sample_for_single_item_dataset(self) -> None:
        context = PipelineContext(dataset=_FakeDataset(1), index=0, rng=self.rng)

        with self.assertLogs("autoware_ml.transforms.base", level="WARNING"):
            secondary = context.sample_secondary()

        self.assertEqual(secondary.io_processing_time, 0.0)

    def test_pre_transform_runs_with_secondary_context(self) -> None:
        dataset = _FakeDataset(2)
        pre_transform = TransformsCompose([_AddTenTransform()])
        context = PipelineContext(dataset=dataset, index=0, rng=self.rng)

        secondary = context.sample_secondary(pre_transform=pre_transform)

        # Sample 1 (the only other index) materialized through the pre-transform.
        self.assertEqual(secondary.io_processing_time, 11.0)
        self.assertEqual(dataset.apply_calls, [(1, pre_transform)])

    def test_secondary_context_shares_rng(self) -> None:
        capture = _CaptureRngTransform()
        context = PipelineContext(dataset=_FakeDataset(2), index=0, rng=self.rng)

        context.sample_secondary(pre_transform=TransformsCompose([capture]))

        self.assertEqual(len(capture.seen), 1)
        self.assertIs(capture.seen[0], self.rng)


if __name__ == "__main__":
    unittest.main()
