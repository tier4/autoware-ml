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

"""Tests for the augmentation a pipeline gives the frames of a stream."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
import unittest

import numpy as np

from autoware_ml.transforms.base import BaseTransform, TransformsCompose, random_source


class _Draw(BaseTransform):
    """Append ``count`` uniform draws to the sample, drawn for its stream or for it alone."""

    stream_consistent = True

    def __init__(self, count: int = 1, probability: float | None = None) -> None:
        super().__init__(probability=probability)
        self.count = count

    def sample_parameters(
        self, random_state: np.random.RandomState | None = None
    ) -> Sequence[float]:
        return random_source(random_state).uniform(size=self.count).tolist()

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        model_gt_sample.draws.append(self.sample_parameters())
        return model_gt_sample

    def transform_in_stream(
        self, model_gt_sample: SimpleNamespace, random_state: np.random.RandomState
    ) -> SimpleNamespace:
        model_gt_sample.draws.append(self.sample_parameters(random_state))
        return model_gt_sample


class _Noise(BaseTransform):
    """Append one draw of its own to every sample, as a jitter that is not stream consistent."""

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        model_gt_sample.draws.append([float(np.random.uniform())])
        return model_gt_sample


class _Unfinished(BaseTransform):
    """A transform that claims to be stream consistent without transform_in_stream."""

    stream_consistent = True

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        return model_gt_sample


class TestStreamAugmentation(unittest.TestCase):
    """The augmentation a pipeline gives the frames of a stream."""

    def draws(
        self, pipeline: TransformsCompose, stream_seed: Sequence[int] | None = None
    ) -> Sequence[Sequence[float]]:
        """The draws every transform of the pipeline applies to a fresh sample of a stream."""
        return pipeline(SimpleNamespace(draws=[]), stream_seed=stream_seed).draws

    def test_frames_of_one_stream_get_the_same_augmentation_and_streams_differ(self) -> None:
        """
        Input: a pipeline of two transforms drawing 3 and 2 values, two frames transformed
        with the seed of one stream and one frame with the seed of another.
        Expected: every frame called with the seed of a stream draws the values of the
        stream, and the two transforms draw values of their own.
        Check: both frames of the first stream record the same 3 and 2 values, the frame of
        the other stream records different ones, and the 2 values of the second transform
        are not the first 2 of the first transform.
        """
        pipeline = TransformsCompose([_Draw(3), _Draw(2)])

        first = self.draws(pipeline, (0, 0, 0))
        second = self.draws(pipeline, (0, 0, 0))
        other = self.draws(pipeline, (0, 0, 1))

        self.assertEqual([len(draw) for draw in first], [3, 2])
        self.assertEqual(second, first)
        self.assertNotEqual(other, first)
        self.assertNotEqual(first[1], first[0][:2])

    def test_a_stream_gets_its_augmentation_while_other_streams_are_read(self) -> None:
        """
        Input: a pipeline of one transform reading the frames of two streams in turn, three
        frames of each, as a dataloader worker reads the lanes of a batch.
        Expected: a frame is augmented by its own stream, whatever stream was read before.
        Check: the three frames of either stream record one draw, and the two streams
        record different ones.
        """
        pipeline = TransformsCompose([_Draw(3)])

        frames: dict[int, list[Sequence[Sequence[float]]]] = {0: [], 1: []}
        for _ in range(3):
            for scene in (0, 1):
                frames[scene].append(self.draws(pipeline, (0, 0, scene)))

        self.assertEqual(frames[0], [frames[0][0] for _ in range(3)])
        self.assertEqual(frames[1], [frames[1][0] for _ in range(3)])
        self.assertNotEqual(frames[0][0], frames[1][0])

    def test_a_stream_is_drawn_alike_in_every_worker(self) -> None:
        """
        Input: two pipelines built alike, as two dataloader workers hold them, each reading
        a frame of the same stream, the second one after frames of 300 other streams.
        Expected: the draw depends only on the seed of the stream, a pipeline keeps nothing
        of the streams it read.
        Check: both pipelines record the same values.
        """
        pipeline, other_worker = TransformsCompose([_Draw(3)]), TransformsCompose([_Draw(3)])

        for scene in range(1, 301):
            self.draws(other_worker, (0, 0, scene))

        self.assertEqual(self.draws(other_worker, (0, 0, 0)), self.draws(pipeline, (0, 0, 0)))

    def test_the_draw_of_a_transform_does_not_depend_on_the_transforms_before_it(self) -> None:
        """
        Input: two pipelines whose second transform draws 2 values, after a first transform
        drawing 3 values in one pipeline and 5 in the other, each reading one stream.
        Expected: every transform draws from a random state of its own, seeded by the stream
        and its position in the pipeline.
        Check: the second transform records the same 2 values in both pipelines.
        """
        short, long = (
            TransformsCompose([_Draw(3), _Draw(2)]),
            TransformsCompose([_Draw(5), _Draw(2)]),
        )

        self.assertEqual(self.draws(short, (0, 0, 0))[1], self.draws(long, (0, 0, 0))[1])

    def test_frames_outside_a_stream_draw_on_their_own(self) -> None:
        """
        Input: a pipeline of one transform drawing 3 values, two samples called without a
        seed before any stream and two after a frame of a stream was transformed.
        Expected: a call without a seed draws its own augmentation, whatever stream the
        pipeline served before it.
        Check: the two samples record different values, before the stream and after it, and
        neither sample after the stream records the values of the stream.
        """
        pipeline = TransformsCompose([_Draw(3)])

        before = [self.draws(pipeline), self.draws(pipeline)]
        in_stream = self.draws(pipeline, (0, 0, 0))
        after = [self.draws(pipeline), self.draws(pipeline)]

        self.assertNotEqual(before[0], before[1])
        self.assertNotEqual(after[0], after[1])
        self.assertNotIn(in_stream, after)

    def test_a_transform_that_is_not_stream_consistent_draws_for_every_frame(self) -> None:
        """
        Input: a pipeline of a stream consistent transform and a jitter that is not, and two
        frames of one stream.
        Expected: the jitter keeps drawing per frame inside the stream.
        Check: the first transform records the same values in both frames and the jitter
        different ones.
        """
        pipeline = TransformsCompose([_Draw(1), _Noise()])

        first = self.draws(pipeline, (0, 0, 0))
        second = self.draws(pipeline, (0, 0, 0))

        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_a_transform_with_a_probability_is_applied_to_the_whole_stream(self) -> None:
        """
        Input: a transform applied with a probability of one half, 40 streams and 5 frames
        of every stream.
        Expected: the probability is drawn from the stream, so it is decided alike for
        every frame of it.
        Check: the 5 frames of a stream all record a draw or all record none, and both
        outcomes occur over the 40 streams.
        """
        pipeline = TransformsCompose([_Draw(1, probability=0.5)])

        outcomes = set()
        for scene in range(40):
            applied = {len(self.draws(pipeline, (0, 0, scene))) for _ in range(5)}
            self.assertEqual(len(applied), 1)
            outcomes.update(applied)

        self.assertEqual(outcomes, {0, 1})

    def test_a_frame_of_a_stream_leaves_the_global_random_state_as_it_was(self) -> None:
        """
        Input: a seeded NumPy random state, read once directly and once after a pipeline
        transformed a frame of a stream.
        Expected: a stream consistent transform draws from the random state of the stream,
        not from the global one.
        Check: the three values drawn after the frame are the ones drawn without it.
        """
        pipeline = TransformsCompose([_Draw(3)])
        np.random.seed(11)
        expected = np.random.uniform(size=3).tolist()

        np.random.seed(11)
        self.draws(pipeline, (0, 0, 0))

        self.assertEqual(np.random.uniform(size=3).tolist(), expected)

    def test_a_stream_consistent_transform_needs_transform_in_stream(self) -> None:
        """
        Input: a transform that sets stream_consistent without overriding
        transform_in_stream, and a frame of a stream.
        Expected: the missing method is reported instead of the frame being augmented on its
        own.
        Check: the call raises a NotImplementedError naming the transform.
        """
        pipeline = TransformsCompose([_Unfinished()])

        with self.assertRaisesRegex(NotImplementedError, "_Unfinished"):
            self.draws(pipeline, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
