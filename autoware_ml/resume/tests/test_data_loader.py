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

"""Unit tests for the resumable training sampler and dataloader."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch
from torch.utils.data import Dataset

from autoware_ml.resume.data_loader import (
    LoaderPosition,
    ResumableDataLoader,
    ResumableDistributedSampler,
)


class _IndexDataset(Dataset):
    """Dataset whose samples are their own indices."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _collate(items: list[int]) -> torch.Tensor:
    """Stack sample indices into one tensor."""
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
    """Resumable loader over an index dataset of ``size`` samples."""
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
    """Iterate one epoch of the loader into plain lists."""
    return [batch.tolist() for batch in loader]


class TestLoaderPosition(unittest.TestCase):
    """Unit tests for LoaderPosition."""

    def test_round_trips_through_the_checkpoint_mapping(self) -> None:
        """Test that the plain mapping keeps the checkpoint format and loads back."""
        position = LoaderPosition(epoch=5, batches_yielded=6, batch_size=4, num_replicas=2)

        state = position.to_dict()

        self.assertEqual(
            state, {"epoch": 5, "batches_yielded": 6, "batch_size": 4, "num_replicas": 2}
        )
        self.assertEqual(LoaderPosition.from_dict(state), position)

    def test_from_dict_requires_every_key(self) -> None:
        """Test that a mapping with a missing field is rejected by name."""
        with self.assertRaisesRegex(KeyError, "batches_yielded"):
            LoaderPosition.from_dict({"epoch": 0})

    def test_batches_to_skip(self) -> None:
        """Test the conversion of a position into batches of another layout."""
        position = LoaderPosition(epoch=0, batches_yielded=3, batch_size=10, num_replicas=2)
        cases = (
            # (batch_size, num_replicas, expected batches)
            (10, 2, 3),
            (5, 2, 6),
            (10, 4, 1),
            (4, 1, 15),
        )
        for batch_size, num_replicas, expected in cases:
            with self.subTest(batch_size=batch_size, num_replicas=num_replicas):
                self.assertEqual(position.batches_to_skip(batch_size, num_replicas), expected)


class TestResumableDistributedSampler(unittest.TestCase):
    """Unit tests for ResumableDistributedSampler."""

    def test_epoch_order_is_a_permutation_that_depends_on_the_epoch(self) -> None:
        """Test that an epoch is a seeded permutation, the same for the same epoch."""
        loader = _loader(23, 4)
        loader.resumable_sampler.set_epoch(0)
        first = [index for batch in _batches(loader) for index in batch]
        loader.resumable_sampler.set_epoch(0)
        again = [index for batch in _batches(loader) for index in batch]
        loader.resumable_sampler.set_epoch(1)
        second = [index for batch in _batches(loader) for index in batch]

        self.assertEqual(sorted(first), list(range(23)))
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertEqual(len(loader), 6)

    def test_sequential_order_when_shuffle_is_off(self) -> None:
        """Test that the order is the dataset order without shuffling."""
        loader = _loader(10, 3, shuffle=False)

        self.assertEqual([index for batch in _batches(loader) for index in batch], list(range(10)))

    def test_weighted_sampler_draws_from_the_weights(self) -> None:
        """Test that samples of zero weight are never drawn."""
        loader = _loader(15, 5, weights=[0.0] * 10 + [1.0] * 5)

        drawn = {index for batch in _batches(loader) for index in batch}

        self.assertLessEqual(drawn, set(range(10, 15)))
        self.assertEqual(len(loader), 3)

    def test_default_seed_is_lightnings_global_seed(self) -> None:
        """Test that the seed defaults to PL_GLOBAL_SEED, and to 0 without it."""
        with mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "17"}):
            self.assertEqual(ResumableDistributedSampler.default_seed(), 17)
            self.assertEqual(ResumableDistributedSampler(_IndexDataset(4)).seed, 17)
        with mock.patch.dict(os.environ):
            os.environ.pop("PL_GLOBAL_SEED", None)
            self.assertEqual(ResumableDistributedSampler.default_seed(), 0)

    def test_single_process_layout_without_ddp(self) -> None:
        """Test that the sampler sees one rank when no process group is initialized."""
        self.assertEqual(ResumableDistributedSampler.distributed_layout(), (1, 0))


class TestResumableDataLoader(unittest.TestCase):
    """Unit tests for ResumableDataLoader."""

    def test_shuffle_with_a_sampler_is_rejected(self) -> None:
        """Test that shuffle belongs to the sampler, not to the loader."""
        dataset = _IndexDataset(4)
        sampler = ResumableDistributedSampler(dataset)

        with self.assertRaisesRegex(ValueError, "shuffle"):
            ResumableDataLoader(dataset, sampler=sampler, batch_size=2, shuffle=True)

    def test_resume_continues_with_exactly_the_remaining_batches(self) -> None:
        """Test that a relaunch finishes the epoch with the batches it still owed."""
        for num_workers in (0, 2):
            for weighted in (False, True):
                with self.subTest(num_workers=num_workers, weighted=weighted):
                    self._check_resume(num_workers, weighted)

    def _check_resume(self, num_workers: int, weighted: bool) -> None:
        """Interrupt an epoch after 6 batches, resume it and compare to the full epoch."""
        weights = [1.0 + (index % 3) for index in range(37)] if weighted else None
        reference = _loader(37, 4, weights=weights, num_workers=num_workers)
        reference.resumable_sampler.set_epoch(5)
        full_epoch = _batches(reference)
        self.assertEqual(len(full_epoch), 10)

        interrupted = _loader(37, 4, weights=weights, num_workers=num_workers)
        interrupted.resumable_sampler.set_epoch(5)
        iterator = iter(interrupted)
        seen = [next(iterator).tolist() for _ in range(6)]
        state = interrupted.state_dict()
        self.assertEqual(
            state, {"epoch": 5, "batches_yielded": 6, "batch_size": 4, "num_replicas": 1}
        )
        del iterator

        relaunched = _loader(37, 4, weights=weights, num_workers=num_workers)
        relaunched.load_state_dict(state)
        # The trainer builds the first iterator before it sets the sampler epoch, so
        # loading the state must already have positioned the sampler on epoch 5.
        self.assertEqual(relaunched.resumable_sampler.epoch, 5)
        remaining = _batches(relaunched)
        self.assertEqual(seen + remaining, full_epoch)
        self.assertEqual(relaunched.batches_yielded, 10)

        # The next epoch is complete again and the skip does not linger.
        relaunched.resumable_sampler.set_epoch(6)
        self.assertEqual(len(_batches(relaunched)), 10)

    def test_stale_position_from_an_earlier_epoch_is_dropped(self) -> None:
        """Test that a position of another epoch does not shorten the current one."""
        loader = _loader(20, 5)
        loader.load_state_dict(
            {"epoch": 2, "batches_yielded": 3, "batch_size": 5, "num_replicas": 1}
        )
        loader.resumable_sampler.set_epoch(3)

        self.assertEqual(len(_batches(loader)), 4)

    def test_changed_batch_size_resumes_at_the_same_sample_position(self) -> None:
        """Test that 30 consumed samples in batches of 10 skip 6 batches of 5."""
        loader = _loader(40, 5)
        loader.resume_from(
            LoaderPosition(epoch=0, batches_yielded=3, batch_size=10, num_replicas=1)
        )

        self.assertEqual(len(_batches(loader)), 8 - 6)

    def test_load_state_dict_requires_every_key(self) -> None:
        """Test that an incomplete loader state is rejected by name."""
        loader = _loader(8, 2)

        with self.assertRaisesRegex(KeyError, "batches_yielded"):
            loader.load_state_dict({"epoch": 0})
