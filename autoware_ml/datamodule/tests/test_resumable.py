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

"""Tests for the resumable training sampler and dataloader."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from autoware_ml.datamodule.resumable import ResumableDataLoader, ResumableDistributedSampler


class _IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _collate(items: list[int]) -> torch.Tensor:
    return torch.tensor(items)


def _loader(
    size: int,
    batch_size: int,
    *,
    shuffle: bool = True,
    weights: list[float] | None = None,
    num_workers: int = 0,
    seed: int = 3,
) -> ResumableDataLoader:
    dataset = _IndexDataset(size)
    sampler = ResumableDistributedSampler(dataset, shuffle=shuffle, weights=weights, seed=seed)
    return ResumableDataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _batches(loader: ResumableDataLoader) -> list[list[int]]:
    return [batch.tolist() for batch in loader]


def test_epoch_order_is_a_permutation_that_depends_on_the_epoch() -> None:
    loader = _loader(23, 4)
    loader.resumable_sampler.set_epoch(0)
    first = [index for batch in _batches(loader) for index in batch]
    loader.resumable_sampler.set_epoch(0)
    again = [index for batch in _batches(loader) for index in batch]
    loader.resumable_sampler.set_epoch(1)
    second = [index for batch in _batches(loader) for index in batch]
    assert sorted(first) == list(range(23))
    assert first == again
    assert first != second
    assert len(loader) == 6


def test_sequential_order_when_shuffle_is_off() -> None:
    loader = _loader(10, 3, shuffle=False)
    assert [index for batch in _batches(loader) for index in batch] == list(range(10))


def test_weighted_sampler_draws_from_the_weights() -> None:
    weights = [0.0] * 10 + [1.0] * 5
    loader = _loader(15, 5, weights=weights)
    drawn = {index for batch in _batches(loader) for index in batch}
    assert drawn <= set(range(10, 15))
    assert len(loader) == 3


def test_shuffle_with_a_sampler_is_rejected() -> None:
    dataset = _IndexDataset(4)
    sampler = ResumableDistributedSampler(dataset)
    with pytest.raises(ValueError, match="shuffle"):
        ResumableDataLoader(dataset, sampler=sampler, batch_size=2, shuffle=True)


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("weighted", [False, True], ids=["uniform", "weighted"])
def test_resume_continues_with_exactly_the_remaining_batches(
    num_workers: int, weighted: bool
) -> None:
    weights = [1.0 + (index % 3) for index in range(37)] if weighted else None
    reference = _loader(37, 4, weights=weights, num_workers=num_workers)
    reference.resumable_sampler.set_epoch(5)
    full_epoch = _batches(reference)
    assert len(full_epoch) == 10

    interrupted = _loader(37, 4, weights=weights, num_workers=num_workers)
    interrupted.resumable_sampler.set_epoch(5)
    seen: list[list[int]] = []
    iterator = iter(interrupted)
    for _ in range(6):
        seen.append(next(iterator).tolist())
    state = interrupted.state_dict()
    assert state == {"epoch": 5, "batches_yielded": 6, "batch_size": 4, "num_replicas": 1}
    del iterator

    relaunched = _loader(37, 4, weights=weights, num_workers=num_workers)
    relaunched.load_state_dict(state)
    # The trainer builds the first iterator before it sets the sampler epoch, so
    # loading the state must already have positioned the sampler on epoch 5.
    assert relaunched.resumable_sampler.epoch == 5
    remaining = _batches(relaunched)
    assert seen + remaining == full_epoch
    assert relaunched.batches_yielded == 10

    # The next epoch is complete again and the skip does not linger.
    relaunched.resumable_sampler.set_epoch(6)
    assert len(_batches(relaunched)) == 10


def test_stale_position_from_an_earlier_epoch_is_dropped() -> None:
    loader = _loader(20, 5)
    loader.load_state_dict({"epoch": 2, "batches_yielded": 3, "batch_size": 5, "num_replicas": 1})
    loader.resumable_sampler.set_epoch(3)
    assert len(_batches(loader)) == 4


def test_changed_batch_size_resumes_at_the_same_sample_position() -> None:
    loader = _loader(40, 5)
    loader.load_state_dict({"epoch": 0, "batches_yielded": 3, "batch_size": 10, "num_replicas": 1})
    # 30 samples consumed with batches of 10 equal 6 batches of 5.
    assert len(_batches(loader)) == 8 - 6


def test_state_dict_requires_every_key() -> None:
    loader = _loader(8, 2)
    with pytest.raises(KeyError, match="batches_yielded"):
        loader.load_state_dict({"epoch": 0})
