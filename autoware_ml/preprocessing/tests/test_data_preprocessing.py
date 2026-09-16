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

"""Unit tests for the DataPreprocessing pipeline wrapper."""

from typing import Any

import pytest

from autoware_ml.preprocessing.base import DataPreprocessing


class _ModeRecorder:
    """Minimal pipeline stage that records the mode it was called with."""

    def __init__(self) -> None:
        self.seen_modes: list[bool] = []

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        self.seen_modes.append(is_training)
        return {"stage_ran": True}


def test_call_forwards_is_training_to_every_layer():
    """The pipeline is not a registered submodule, so the owning model's mode reaches
    the stages only through the explicit is_training argument."""
    first, second = _ModeRecorder(), _ModeRecorder()
    pipeline = DataPreprocessing([first, second])

    pipeline({}, is_training=True)
    pipeline({}, is_training=False)

    assert first.seen_modes == [True, False]
    assert second.seen_modes == [True, False]


def test_call_requires_explicit_is_training():
    """The mode must be stated on every call; forgetting it is an immediate TypeError
    instead of silently running in the wrong mode."""
    pipeline = DataPreprocessing([_ModeRecorder()])

    with pytest.raises(TypeError):
        pipeline({})  # type: ignore[call-arg]


def test_call_merges_layer_outputs_into_batch():
    pipeline = DataPreprocessing([_ModeRecorder()])
    batch = {"points": [1, 2, 3]}

    result = pipeline(batch, is_training=True)

    assert result is batch  # mutated in place and returned for chaining
    assert result["stage_ran"] is True
    assert result["points"] == [1, 2, 3]
