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

"""Training dataloader that resumes exactly where a preempted run stopped.

Lightning restores the loop counters of a mid-epoch checkpoint but not the data
position: it runs the remaining number of batches from a fresh iterator, so the
head of the epoch is seen twice and its tail never. The two classes here close
that gap.

* :class:`ResumableDistributedSampler` draws the order of an epoch from
  ``(seed, epoch)`` alone, so any process can rebuild it, and can skip the first
  samples of the next iteration.
* :class:`ResumableDataLoader` counts the batches it hands out and exposes that
  count through ``state_dict``/``load_state_dict``. Loading a state arms a skip
  that the next ``__iter__`` at the same epoch consumes.

The datamodule stores the loader state in the checkpoint through its own
``state_dict`` hook, which Lightning restores before any dataloader is built, so
the position survives even when the trainer builds the dataloaders early (it does,
to estimate the number of optimizer steps for the scheduler).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def default_sampler_seed() -> int:
    """Return the seed Lightning uses for the distributed samplers it injects."""
    return int(os.environ.get("PL_GLOBAL_SEED", "0"))


class ResumableDistributedSampler(DistributedSampler):
    """Deterministic per epoch sampler that can skip the head of an epoch.

    The order of epoch ``e`` depends only on ``seed + e``: a uniform permutation,
    the identity, or a weighted multinomial draw when ``weights`` are given (the
    repeat factor frame sampling). The epoch is partitioned across ranks the way
    :class:`~torch.utils.data.distributed.DistributedSampler` does, and every
    rank sees the same number of samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        shuffle: bool = True,
        weights: Sequence[float] | None = None,
        replacement: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: Dataset sampled by the dataloader.
            shuffle: Whether to permute the samples of every epoch. Ignored when
                ``weights`` are given, a weighted draw is random by construction.
            weights: Optional per sample sampling weights; when given the epoch is
                a multinomial draw of ``len(dataset)`` samples.
            replacement: Whether the weighted draw samples with replacement.
            seed: Base seed of the epoch order. Defaults to Lightning's global seed.
            drop_last: Whether to drop tail samples instead of padding so that the
                epoch divides evenly across ranks.

        Raises:
            ValueError: Raised for weights of the wrong length or with no mass.
        """
        num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=default_sampler_seed() if seed is None else seed,
            drop_last=drop_last,
        )
        self.weights: torch.Tensor | None = None
        if weights is not None:
            if len(weights) != len(dataset):
                raise ValueError(f"Expected {len(dataset)} sampler weights, got {len(weights)}.")
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            if torch.any(self.weights < 0):
                raise ValueError("Sampler weights must be non-negative.")
            if float(self.weights.sum().item()) <= 0.0:
                raise ValueError("At least one sampler weight must be positive.")
        self.replacement = replacement
        self._skip_samples = 0

    def skip_samples(self, count: int) -> None:
        """Skip the first samples of this rank's share on the next iteration only.

        Args:
            count: Number of samples to skip.
        """
        self._skip_samples = max(0, int(count))

    def epoch_indices(self) -> list[int]:
        """Return this rank's complete sample order for the current epoch."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.weights is not None:
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=self.replacement, generator=generator
            ).tolist()
        else:
            dataset_size = len(self.dataset)  # type: ignore[arg-type]
            if self.shuffle:
                indices = torch.randperm(dataset_size, generator=generator).tolist()
            else:
                indices = list(range(dataset_size))
            if self.drop_last:
                indices = indices[: self.total_size]
            elif indices:
                padding = self.total_size - len(indices)
                if padding > 0:
                    indices += (indices * math.ceil(padding / len(indices)))[:padding]
        return indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        """Yield this rank's sample indices, minus a pending skip."""
        indices = self.epoch_indices()
        skip, self._skip_samples = self._skip_samples, 0
        return iter(indices[skip:])


class ResumableDataLoader(DataLoader):
    """Dataloader whose position in the epoch survives a checkpoint.

    The loader counts the batches it has handed to the training loop; a
    checkpoint written after batch ``k`` records ``k``. On resume the count is
    turned into a sample skip on the sampler, so the interrupted epoch continues
    with batch ``k + 1`` and finishes with exactly the batches it still owed.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        sampler: ResumableDistributedSampler,
        batch_size: int,
        **kwargs: Any,
    ) -> None:
        """Initialize the loader.

        Args:
            dataset: Dataset to load.
            sampler: The resumable sampler drawing the order.
            batch_size: Samples per batch.
            **kwargs: Remaining :class:`~torch.utils.data.DataLoader` arguments.

        Raises:
            TypeError: Raised when the sampler is not resumable.
        """
        if not isinstance(sampler, ResumableDistributedSampler):
            raise TypeError(
                "ResumableDataLoader needs a ResumableDistributedSampler, got "
                f"{type(sampler).__name__}."
            )
        if kwargs.pop("shuffle", False):
            raise ValueError("Pass shuffle to the sampler, not to the ResumableDataLoader.")
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
        self._batches_yielded = 0
        self._resume_state: dict[str, int] | None = None

    @property
    def resumable_sampler(self) -> ResumableDistributedSampler:
        """Return the sampler drawing the epoch order."""
        return self.sampler  # type: ignore[return-value]

    @property
    def batches_yielded(self) -> int:
        """Return the number of batches handed out so far in the current epoch."""
        return self._batches_yielded

    def state_dict(self) -> dict[str, int]:
        """Return the position of the loader in the current epoch."""
        sampler = self.resumable_sampler
        return {
            "epoch": int(sampler.epoch),
            "batches_yielded": int(self._batches_yielded),
            "batch_size": int(self.batch_size or 1),
            "num_replicas": int(sampler.num_replicas),
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Arm the loader to continue from a saved position.

        The sampler is moved to the saved epoch right away, because the trainer
        builds the first iterator before it sets the epoch on the samplers.

        Args:
            state_dict: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: Raised when the state misses a required key.
        """
        required = ("epoch", "batches_yielded", "batch_size", "num_replicas")
        missing = [key for key in required if key not in state_dict]
        if missing:
            raise KeyError(f"Resumable dataloader state misses {missing}.")
        self._resume_state = {key: int(state_dict[key]) for key in required}
        self.resumable_sampler.set_epoch(self._resume_state["epoch"])

    def _batches_to_skip(self, state: dict[str, int]) -> int:
        """Translate a saved position into batches of this loader."""
        batch_size = int(self.batch_size or 1)
        num_replicas = self.resumable_sampler.num_replicas
        if state["batch_size"] == batch_size and state["num_replicas"] == num_replicas:
            return state["batches_yielded"]
        consumed = state["batches_yielded"] * state["batch_size"] * state["num_replicas"]
        skip = consumed // (batch_size * num_replicas)
        logger.warning(
            "Resuming a dataloader saved with batch_size=%d on %d ranks into batch_size=%d on "
            "%d ranks; continuing from batch %d of the epoch instead of %d.",
            state["batch_size"],
            state["num_replicas"],
            batch_size,
            num_replicas,
            skip,
            state["batches_yielded"],
        )
        return skip

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        """Iterate the epoch, skipping the batches a loaded state already consumed."""
        start = 0
        state = self._resume_state
        if state is not None:
            if state["epoch"] == self.resumable_sampler.epoch:
                start = self._batches_to_skip(state)
                self.resumable_sampler.skip_samples(start * int(self.batch_size or 1))
                logger.info(
                    "Resuming the training dataloader at batch %d of epoch %d.",
                    start,
                    state["epoch"],
                )
            else:
                logger.info(
                    "Dropping a dataloader position saved at epoch %d, iterating epoch %d.",
                    state["epoch"],
                    self.resumable_sampler.epoch,
                )
            self._resume_state = None
        self._batches_yielded = start
        for batch in super().__iter__():
            self._batches_yielded += 1
            yield batch
