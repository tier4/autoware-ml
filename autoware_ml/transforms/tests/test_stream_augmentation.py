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

from types import SimpleNamespace

import numpy as np
import pytest

from autoware_ml.transforms.base import BaseTransform, StreamParameters, TransformsCompose


class _Draw(BaseTransform):
    """Append ``count`` uniform draws to the sample, as an augmentation reads its parameters."""

    def __init__(self, count: int = 1, probability: float | None = None) -> None:
        super().__init__(probability=probability)
        self.count = count

    def sample_parameters(self) -> list[float]:
        return np.random.uniform(size=self.count).tolist()

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        return self.transform_with_parameters(model_gt_sample, self.sample_parameters())

    def transform_with_parameters(
        self, model_gt_sample: SimpleNamespace, parameters: list[float]
    ) -> SimpleNamespace:
        model_gt_sample.draws.append(parameters)
        return model_gt_sample


class _Noise(BaseTransform):
    """Append one draw of its own to every sample, as a jitter that no stream shares."""

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        model_gt_sample.draws.append([float(np.random.uniform())])
        return model_gt_sample


def _draws(pipeline: TransformsCompose, stream_parameters: tuple | None) -> list[list[float]]:
    return pipeline(SimpleNamespace(draws=[]), stream_parameters).draws


def test_frames_of_one_stream_get_the_drawn_augmentation_and_streams_differ() -> None:
    pipeline = TransformsCompose([_Draw(3), _Draw(2)])
    stream, other_stream = (
        pipeline.sample_stream_parameters(),
        pipeline.sample_stream_parameters(),
    )

    first, second, other = (
        _draws(pipeline, stream),
        _draws(pipeline, stream),
        _draws(pipeline, other_stream),
    )

    assert first == second == [stream[0].values, stream[1].values]
    assert first != other


def test_frames_outside_a_stream_draw_on_their_own() -> None:
    pipeline = TransformsCompose([_Draw(3)])

    assert _draws(pipeline, None) != _draws(pipeline, None)


def test_a_transform_without_stream_parameters_draws_for_every_frame_of_a_stream() -> None:
    pipeline = TransformsCompose([_Draw(1), _Noise()])
    stream = pipeline.sample_stream_parameters()

    first, second = _draws(pipeline, stream), _draws(pipeline, stream)

    assert stream[1] is None
    assert first[0] == second[0]
    assert first[1] != second[1]


def test_a_transform_applied_with_a_probability_is_applied_to_the_whole_stream() -> None:
    pipeline = TransformsCompose([_Draw(1, probability=0.5)])
    streams = [pipeline.sample_stream_parameters() for _ in range(40)]

    assert {stream[0].apply for stream in streams} == {True, False}
    for stream in streams:
        applied = [len(_draws(pipeline, stream)) for _ in range(5)]
        assert applied == [int(stream[0].apply)] * 5


def test_a_skipped_transform_draws_no_parameters() -> None:
    assert _Draw(1, probability=0.0).sample_stream_parameters() == StreamParameters(False, None)


def test_the_parameters_of_another_pipeline_are_rejected() -> None:
    pipeline = TransformsCompose([_Draw(1), _Draw(1)])

    with pytest.raises(ValueError, match="pipeline of 2"):
        _draws(pipeline, TransformsCompose([_Draw(1)]).sample_stream_parameters())
