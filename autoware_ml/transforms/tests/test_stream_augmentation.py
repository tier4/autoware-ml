# Copyright 2025 TIER IV, Inc.
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

"""Tests for the augmentation a pipeline draws once per stream."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
import unittest

import numpy as np

from autoware_ml.transforms.base import BaseTransform, StreamParameters, TransformsCompose


class _Draw(BaseTransform):
    """Append ``count`` uniform draws to the sample, as an augmentation reads its parameters."""

    def __init__(self, count: int = 1, probability: float | None = None) -> None:
        super().__init__(probability=probability)
        self.count = count

    def sample_parameters(self) -> Sequence[float]:
        return np.random.uniform(size=self.count).tolist()

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        return self.transform_with_parameters(model_gt_sample, self.sample_parameters())

    def transform_with_parameters(
        self, model_gt_sample: SimpleNamespace, parameters: Sequence[float]
    ) -> SimpleNamespace:
        model_gt_sample.draws.append(parameters)
        return model_gt_sample


class _Noise(BaseTransform):
    """Append one draw of its own to every sample, as a jitter that no stream shares."""

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        model_gt_sample.draws.append([float(np.random.uniform())])
        return model_gt_sample


class TestStreamAugmentation(unittest.TestCase):
    """The augmentation a pipeline draws once for a stream and applies to its frames."""

    def draws(
        self, pipeline: TransformsCompose, stream_parameters: tuple | None
    ) -> Sequence[Sequence[float]]:
        """The draws every transform of the pipeline applies to a fresh sample."""
        return pipeline(SimpleNamespace(draws=[]), stream_parameters).draws

    def test_frames_of_one_stream_get_the_drawn_augmentation_and_streams_differ(self) -> None:
        """
        Input: a pipeline of two transforms drawing 3 and 2 values, the parameters of two
        streams drawn from it, and two frames of the first stream and one of the second.
        Expected: every frame of a stream is transformed with the values drawn for it.
        Check: both frames of the first stream record its drawn values, and the frame of the
        other stream records different ones.
        """
        pipeline = TransformsCompose([_Draw(3), _Draw(2)])
        stream = pipeline.sample_stream_parameters()
        other_stream = pipeline.sample_stream_parameters()

        first = self.draws(pipeline, stream)
        second = self.draws(pipeline, stream)
        other = self.draws(pipeline, other_stream)

        self.assertEqual(first, [stream[0].values, stream[1].values])
        self.assertEqual(second, first)
        self.assertNotEqual(other, first)

    def test_frames_outside_a_stream_draw_on_their_own(self) -> None:
        """
        Input: a pipeline of one transform drawing 3 values, called twice without stream
        parameters.
        Expected: every sample draws its own augmentation.
        Check: the two samples record different values.
        """
        pipeline = TransformsCompose([_Draw(3)])

        self.assertNotEqual(self.draws(pipeline, None), self.draws(pipeline, None))

    def test_a_transform_without_stream_parameters_draws_for_every_frame(self) -> None:
        """
        Input: a pipeline of a transform that shares its draw with a stream and a jitter that
        does not implement sample_parameters, and two frames of one stream.
        Expected: the stream holds no parameters for the jitter, which keeps drawing per frame.
        Check: the stream entry of the jitter is None, the first transform records the same
        values in both frames and the jitter different ones.
        """
        pipeline = TransformsCompose([_Draw(1), _Noise()])
        stream = pipeline.sample_stream_parameters()

        first = self.draws(pipeline, stream)
        second = self.draws(pipeline, stream)

        self.assertIsNone(stream[1])
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_a_transform_with_a_probability_is_applied_to_the_whole_stream(self) -> None:
        """
        Input: a transform applied with a probability of one half, 40 streams drawn from it
        and 5 frames of every stream.
        Expected: the probability is decided once per stream, not per frame.
        Check: some streams apply the transform and some skip it, and the 5 frames of a
        stream all record a draw or all record none.
        """
        pipeline = TransformsCompose([_Draw(1, probability=0.5)])
        streams = [pipeline.sample_stream_parameters() for _ in range(40)]

        self.assertEqual({stream[0].apply for stream in streams}, {True, False})
        for stream in streams:
            applied = {len(self.draws(pipeline, stream)) for _ in range(5)}
            self.assertEqual(applied, {int(stream[0].apply)})

    def test_a_skipped_transform_draws_no_parameters(self) -> None:
        """
        Input: a transform with a probability of zero.
        Expected: the stream skips the transform without drawing its values.
        Check: the stream parameters are apply False with no values.
        """
        self.assertEqual(
            _Draw(1, probability=0.0).sample_stream_parameters(), StreamParameters(False, None)
        )

    def test_the_parameters_of_another_pipeline_are_rejected(self) -> None:
        """
        Input: a pipeline of two transforms given the stream parameters of a pipeline of one.
        Expected: parameters that cannot be matched to the transforms are a caller error.
        Check: the pipeline raises a ValueError naming its length of 2.
        """
        pipeline = TransformsCompose([_Draw(1), _Draw(1)])
        other_parameters = TransformsCompose([_Draw(1)]).sample_stream_parameters()

        with self.assertRaisesRegex(ValueError, "pipeline of 2"):
            self.draws(pipeline, other_parameters)


if __name__ == "__main__":
    unittest.main()
