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

"""Tests for the resumable training dataloader: the resumed epoch continues after the yielded
batches with any worker setup, and a state of another epoch is ignored."""

import pytest
from torch.utils.data import DistributedSampler

from autoware_ml.datamodule.resumable_data_loader import ResumableDataLoader

_DATASET = list(range(23))
_WORKER_SETUPS = [
    pytest.param({"num_workers": 0}, id="main-process"),
    pytest.param({"num_workers": 2}, id="workers"),
    pytest.param({"num_workers": 2, "persistent_workers": True}, id="persistent-workers"),
]


def _loader(**kwargs) -> ResumableDataLoader:
    sampler = DistributedSampler(_DATASET, num_replicas=1, rank=0, shuffle=True, seed=7)
    return ResumableDataLoader(_DATASET, batch_size=4, sampler=sampler, **kwargs)


def _epoch(loader: ResumableDataLoader, epoch: int) -> list[list[int]]:
    loader.sampler.set_epoch(epoch)
    return [batch.tolist() for batch in loader]


@pytest.mark.parametrize("worker_setup", _WORKER_SETUPS)
def test_resumed_epoch_continues_after_the_yielded_batches(worker_setup: dict) -> None:
    reference = _epoch(_loader(), 1)
    interrupted = _loader(**worker_setup)
    interrupted.sampler.set_epoch(1)
    iterator = iter(interrupted)
    for _ in range(3):
        next(iterator)
    state = interrupted.state_dict()

    resumed = _loader(**worker_setup)
    resumed.load_state_dict(state)

    assert state == {"epoch": 1, "batches": 3}
    assert len(resumed) == len(reference)
    assert _epoch(resumed, 1) == reference[3:]
    assert resumed.state_dict() == {"epoch": 1, "batches": len(reference)}
    assert _epoch(resumed, 2) == _epoch(_loader(), 2)


def test_state_of_another_epoch_is_ignored() -> None:
    resumed = _loader()
    resumed.load_state_dict({"epoch": 1, "batches": 6})

    assert _epoch(resumed, 2) == _epoch(_loader(), 2)
