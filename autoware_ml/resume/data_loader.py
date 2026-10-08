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

* :class:`LoaderPosition` is the position of the loader in an epoch, as stored in
  the checkpoint.
* :class:`ResumableDistributedSampler` draws the order of an epoch from
  ``(seed, epoch)`` alone, so any process can rebuild it, and can skip the first
  samples of the next iteration.
* :class:`ResumableDataLoader` counts the batches it hands out and exposes that
  count through ``state_dict``/``load_state_dict``. Loading a state arms a skip
  that the next ``__iter__`` at the same epoch consumes.

The datamodule stores the loader state in the checkpoint through its own
``state_dict`` hook (:class:`ResumableTrainLoaderMixin`), which Lightning restores before any dataloader is built, so
the position survives even when the trainer builds the dataloaders early (it does,
to estimate the number of optimizer steps for the scheduler).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoaderPosition:
    """Position of the training dataloader in an epoch.

    Attributes:
        epoch: Epoch the position belongs to.
        batches_yielded: Batches handed to the training loop in that epoch.
        batch_size: Samples per batch of the loader that saved the position.
        num_replicas: Number of ranks of the run that saved the position.
    """

    epoch: int
    batches_yielded: int
    batch_size: int
    num_replicas: int

    def to_dict(self) -> dict[str, int]:
        """Return the position as the plain mapping stored in the checkpoint."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, state_dict: Mapping[str, Any]) -> LoaderPosition:
        """Build a position from the mapping stored in the checkpoint.

        Args:
            state_dict: Mapping produced by :meth:`to_dict`.

        Returns:
            The position.

        Raises:
            KeyError: Raised when the mapping misses a field.
        """
        names = [field.name for field in dataclasses.fields(cls)]
        missing = [name for name in names if name not in state_dict]
        if missing:
            raise KeyError(f"Resumable dataloader state misses {missing}.")
        return cls(**{name: int(state_dict[name]) for name in names})

    def has_layout(self, batch_size: int, num_replicas: int) -> bool:
        """Return whether the position was saved with this batch size and rank count.

        Args:
            batch_size: Samples per batch of the resuming loader.
            num_replicas: Number of ranks of the resuming run.

        Returns:
            ``True`` when both match the saving run.
        """
        return self.batch_size == batch_size and self.num_replicas == num_replicas

    def batches_to_skip(self, batch_size: int, num_replicas: int) -> int:
        """Translate the position into batches of a loader with the given layout.

        A changed batch size or rank count is converted through the number of
        samples the whole run consumed, rounded down to a full batch.

        Args:
            batch_size: Samples per batch of the resuming loader.
            num_replicas: Number of ranks of the resuming run.

        Returns:
            Number of batches the resuming loader skips.
        """
        if self.has_layout(batch_size, num_replicas):
            return self.batches_yielded
        consumed = self.batches_yielded * self.batch_size * self.num_replicas
        return consumed // (batch_size * num_replicas)


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
        num_replicas, rank = self.distributed_layout()
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=self.default_seed() if seed is None else seed,
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

    @staticmethod
    def default_seed() -> int:
        """Return the seed Lightning uses for the distributed samplers it injects."""
        return int(os.environ.get("PL_GLOBAL_SEED", "0"))

    @staticmethod
    def distributed_layout() -> tuple[int, int]:
        """Return the number of ranks and this process's rank, ``(1, 0)`` without DDP."""
        if not (dist.is_available() and dist.is_initialized()):
            return 1, 0
        return dist.get_world_size(), dist.get_rank()

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
        self._resume_position: LoaderPosition | None = None

    @property
    def resumable_sampler(self) -> ResumableDistributedSampler:
        """Return the sampler drawing the epoch order."""
        return self.sampler  # type: ignore[return-value]

    @property
    def batches_yielded(self) -> int:
        """Return the number of batches handed out so far in the current epoch."""
        return self._batches_yielded

    @property
    def samples_per_batch(self) -> int:
        """Return the batch size, 1 for a loader without automatic batching."""
        return int(self.batch_size or 1)

    def position(self) -> LoaderPosition:
        """Return the position of the loader in the current epoch."""
        return LoaderPosition(
            epoch=int(self.resumable_sampler.epoch),
            batches_yielded=int(self._batches_yielded),
            batch_size=self.samples_per_batch,
            num_replicas=int(self.resumable_sampler.num_replicas),
        )

    def resume_from(self, position: LoaderPosition) -> None:
        """Arm the loader to continue from a saved position.

        The sampler is moved to the saved epoch right away, because the trainer
        builds the first iterator before it sets the epoch on the samplers.

        Args:
            position: Position saved by an earlier run.
        """
        self._resume_position = position
        self.resumable_sampler.set_epoch(position.epoch)

    def state_dict(self) -> dict[str, int]:
        """Return the position of the loader in the current epoch as a plain mapping."""
        return self.position().to_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Arm the loader to continue from a saved position mapping.

        Args:
            state_dict: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: Raised when the state misses a required key.
        """
        self.resume_from(LoaderPosition.from_dict(state_dict))

    def _consume_resume_position(self) -> int:
        """Apply a pending resume position to the sampler, once.

        Returns:
            Number of batches of the epoch the position already consumed, 0 when no
            position is pending or it belongs to another epoch.
        """
        position, self._resume_position = self._resume_position, None
        if position is None:
            return 0
        epoch = self.resumable_sampler.epoch
        if position.epoch != epoch:
            logger.info(
                "Dropping a dataloader position saved at epoch %d, iterating epoch %d.",
                position.epoch,
                epoch,
            )
            return 0
        batch_size = self.samples_per_batch
        num_replicas = self.resumable_sampler.num_replicas
        start = position.batches_to_skip(batch_size, num_replicas)
        if not position.has_layout(batch_size, num_replicas):
            logger.warning(
                "Resuming a dataloader saved with batch_size=%d on %d ranks into batch_size=%d "
                "on %d ranks; continuing from batch %d of the epoch instead of %d.",
                position.batch_size,
                position.num_replicas,
                batch_size,
                num_replicas,
                start,
                position.batches_yielded,
            )
        self.resumable_sampler.skip_samples(start * batch_size)
        logger.info("Resuming the training dataloader at batch %d of epoch %d.", start, epoch)
        return start

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        """Iterate the epoch, skipping the batches a loaded position already consumed."""
        self._batches_yielded = self._consume_resume_position()
        for batch in super().__iter__():
            self._batches_yielded += 1
            yield batch


class ResumableTrainLoaderMixin:
    """Datamodule hooks that carry the training loader position through a checkpoint.

    Lightning restores the datamodule state before any dataloader is built, so a
    loaded position is kept until :meth:`_attach_train_loader` hands it to the
    training loader. Mix it into a :class:`~lightning.LightningDataModule` before
    the Lightning base class.
    """

    _train_loader: ResumableDataLoader | None = None
    _train_loader_state: LoaderPosition | None = None

    def _attach_train_loader(self, loader: ResumableDataLoader) -> ResumableDataLoader:
        """Track the training loader and arm it with a loaded position.

        Args:
            loader: Freshly built training loader.

        Returns:
            The same loader.
        """
        if self._train_loader_state is not None:
            loader.resume_from(self._train_loader_state)
        self._train_loader = loader
        return loader

    def state_dict(self) -> dict[str, Any]:
        """Return the training loader position for the checkpoint.

        Returns:
            The loader state, empty when no resumable training loader was built.
        """
        if self._train_loader is None:
            return {}
        return {"train_dataloader": self._train_loader.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Remember the training loader position of a resumed checkpoint.

        Args:
            state_dict: Mapping produced by :meth:`state_dict`.
        """
        loader_state = state_dict.get("train_dataloader")
        self._train_loader_state = (
            None if loader_state is None else LoaderPosition.from_dict(loader_state)
        )
        if self._train_loader is not None and self._train_loader_state is not None:
            self._train_loader.resume_from(self._train_loader_state)
