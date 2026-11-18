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

"""Tests for the augmentation the transforms of a pipeline keep for a stream."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
import unittest

import numpy as np

from autoware_ml.transforms.base import BaseTransform, TransformsCompose


class _Draw(BaseTransform):
    """Append ``count`` uniform draws to the sample, kept for a stream or drawn per sample."""

    def __init__(self, count: int = 1, probability: float | None = None) -> None:
        super().__init__(probability=probability)
        self.count = count
        self.parameters: Sequence[float] | None = None
        # Every draw made, shared with the copies a pipeline keeps per stream
        self.sampled: list[Sequence[float]] = []

    def sample_parameters(self) -> Sequence[float]:
        self.sampled.append(np.random.uniform(size=self.count).tolist())
        return self.sampled[-1]

    def draw_parameters(self) -> bool:
        self.parameters = self.sample_parameters()
        return True

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        parameters = self.parameters if self.parameters is not None else self.sample_parameters()
        return self.transform_with_parameters(model_gt_sample, parameters)

    def transform_with_parameters(
        self, model_gt_sample: SimpleNamespace, parameters: Sequence[float]
    ) -> SimpleNamespace:
        model_gt_sample.draws.append(parameters)
        return model_gt_sample


class _Noise(BaseTransform):
    """Append one draw of its own to every sample, as a jitter that keeps no parameters."""

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        model_gt_sample.draws.append([float(np.random.uniform())])
        return model_gt_sample


class TestStreamAugmentation(unittest.TestCase):
    """The augmentation the transforms of a pipeline keep for the frames of a stream."""

    def draws(self, pipeline: TransformsCompose) -> Sequence[Sequence[float]]:
        """The draws every transform of the pipeline applies to a fresh sample."""
        return pipeline(SimpleNamespace(draws=[])).draws

    def test_frames_of_one_stream_get_the_kept_augmentation_and_streams_differ(self) -> None:
        """
        Input: a pipeline of two transforms drawing 3 and 2 values, two frames transformed
        after the draw of one stream and one frame after the draw of another.
        Expected: the transforms keep the values drawn for a stream and apply them to every
        frame until another stream is drawn.
        Check: both frames of the first stream record the kept values, and the frame of the
        other stream records different ones.
        """
        pipeline = TransformsCompose([_Draw(3), _Draw(2)])

        pipeline.keep_stream_parameters((0, 0, 0))
        kept = [transform.parameters for transform in pipeline.active_pipeline]
        first = self.draws(pipeline)
        second = self.draws(pipeline)
        pipeline.keep_stream_parameters((0, 0, 1))
        other = self.draws(pipeline)

        self.assertEqual(first, kept)
        self.assertEqual(second, kept)
        self.assertNotEqual(other, kept)

    def test_a_stream_keeps_its_parameters_while_other_streams_are_read(self) -> None:
        """
        Input: a pipeline of one transform reading the frames of two streams in turn, three
        frames of each, as a dataloader worker reads the lanes of a batch.
        Expected: the parameters of a stream are drawn once, with its first frame, and kept
        for its later frames instead of being drawn again.
        Check: the transform drew twice in all, once per stream, and the three frames of
        either stream record the one draw of their stream.
        """
        transform = _Draw(3)
        pipeline = TransformsCompose([transform])

        frames: dict[int, list[Sequence[Sequence[float]]]] = {0: [], 1: []}
        for _ in range(3):
            for scene in (0, 1):
                pipeline.keep_stream_parameters((0, 0, scene))
                frames[scene].append(self.draws(pipeline))

        self.assertEqual(len(transform.sampled), 2)
        self.assertEqual(frames[0], [[transform.sampled[0]] for _ in range(3)])
        self.assertEqual(frames[1], [[transform.sampled[1]] for _ in range(3)])
        self.assertNotEqual(transform.sampled[0], transform.sampled[1])

    def test_a_stream_is_drawn_alike_in_every_worker(self) -> None:
        """
        Input: two pipelines built alike, as two dataloader workers hold them, each reading
        a frame of the same stream.
        Expected: the draw depends only on the seed of the stream.
        Check: both pipelines record the same values.
        """
        pipeline, other_worker = TransformsCompose([_Draw(3)]), TransformsCompose([_Draw(3)])

        pipeline.keep_stream_parameters((0, 0, 0))
        other_worker.keep_stream_parameters((0, 0, 0))

        self.assertEqual(self.draws(other_worker), self.draws(pipeline))

    def test_a_dropped_stream_is_drawn_with_the_same_values_again(self) -> None:
        """
        Input: a pipeline keeping at most two streams, reading stream 0, then two other
        streams, then stream 0 again.
        Expected: the least recently read stream is dropped, and drawn from its seed again
        when it comes back.
        Check: the pipeline keeps two streams, and stream 0 records the same values before
        and after it was dropped.
        """
        pipeline = TransformsCompose([_Draw(3)])
        pipeline.MAX_KEPT_STREAMS = 2

        pipeline.keep_stream_parameters((0, 0, 0))
        first = self.draws(pipeline)
        pipeline.keep_stream_parameters((0, 0, 1))
        pipeline.keep_stream_parameters((0, 0, 2))
        kept_streams = len(pipeline._stream_pipelines)
        pipeline.keep_stream_parameters((0, 0, 0))

        self.assertEqual(kept_streams, 2)
        self.assertEqual(self.draws(pipeline), first)

    def test_frames_outside_a_stream_draw_on_their_own(self) -> None:
        """
        Input: a pipeline of one transform drawing 3 values, two samples before any stream
        and two samples after the parameters of a stream were cleared.
        Expected: without kept parameters every sample draws its own augmentation.
        Check: the two samples record different values, before the stream and after it.
        """
        pipeline = TransformsCompose([_Draw(3)])

        before = [self.draws(pipeline), self.draws(pipeline)]
        pipeline.keep_stream_parameters((0, 0, 0))
        pipeline.clear_stream_parameters()
        after = [self.draws(pipeline), self.draws(pipeline)]

        self.assertNotEqual(before[0], before[1])
        self.assertNotEqual(after[0], after[1])

    def test_a_transform_that_keeps_no_parameters_draws_for_every_frame(self) -> None:
        """
        Input: a pipeline of a transform that keeps its draw and a jitter that keeps none,
        and two frames of one stream.
        Expected: the jitter keeps drawing per frame inside the stream.
        Check: the first transform records the same values in both frames and the jitter
        different ones.
        """
        pipeline = TransformsCompose([_Draw(1), _Noise()])
        pipeline.keep_stream_parameters((0, 0, 0))

        first = self.draws(pipeline)
        second = self.draws(pipeline)

        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_a_transform_with_a_probability_is_applied_to_the_whole_stream(self) -> None:
        """
        Input: a transform applied with a probability of one half, 40 streams and 5 frames
        of every stream.
        Expected: the probability is decided once per stream, not per frame.
        Check: the 5 frames of a stream all record a draw or all record none, and both
        outcomes occur over the 40 streams.
        """
        pipeline = TransformsCompose([_Draw(1, probability=0.5)])

        outcomes = set()
        for scene in range(40):
            pipeline.keep_stream_parameters((0, 0, scene))
            applied = {len(self.draws(pipeline)) for _ in range(5)}
            self.assertEqual(len(applied), 1)
            outcomes.update(applied)

        self.assertEqual(outcomes, {0, 1})

    def test_drawing_the_augmentation_leaves_the_random_state_as_it_was(self) -> None:
        """
        Input: a seeded NumPy random state, read once directly and once after a pipeline
        drew the augmentation of a stream.
        Expected: the pipeline restores the random state it seeded for its draws.
        Check: the three values drawn after the stream are the ones drawn without it.
        """
        pipeline = TransformsCompose([_Draw(3)])
        np.random.seed(11)
        expected = np.random.uniform(size=3).tolist()

        np.random.seed(11)
        pipeline.keep_stream_parameters((0, 0, 0))

        self.assertEqual(np.random.uniform(size=3).tolist(), expected)


if __name__ == "__main__":
    unittest.main()
