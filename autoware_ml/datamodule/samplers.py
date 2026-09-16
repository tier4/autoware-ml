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

"""Data samplers shared by Autoware-ML datamodules."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class DistributedWeightedRandomSampler(DistributedSampler):
    """Weighted random sampler that partitions one sampled epoch across ranks."""

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        *,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: Dataset sampled by the dataloader.
            weights: Per-sample non-negative sampling weights.
            replacement: Whether to sample indices with replacement.
            seed: Base seed used with ``set_epoch`` for deterministic shuffling.
            drop_last: Whether to drop tail samples when dataset length is not
                divisible by world size.
        """
        if len(weights) != len(dataset):
            raise ValueError(f"Expected {len(dataset)} sampler weights, got {len(weights)}.")
        num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if torch.any(self.weights < 0):
            raise ValueError("Sampler weights must be non-negative.")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("At least one sampler weight must be positive.")
        self.replacement = replacement

    def __iter__(self):
        """Yield the weighted sample indices for this rank."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


class GroupStreamingSampler(Sampler[int]):
    """Interleave scene-contiguous frame indices across dataloader lanes.

    Each of the ``batch_size`` dataloader lanes walks whole scenes frame by
    frame, so consecutive iterations feed consecutive frames of the same scene
    into the same batch position. Stateful temporal models can therefore carry
    per-lane memory across iterations, with scene changes signalled by the
    dataset's ``prev_exists`` metadata. Scenes are partitioned round-robin over
    distributed ranks and each rank assigns every scene to its currently
    shortest lane, which keeps lane lengths close to each other.

    Epoch length is fixed when ``shuffle`` is true (training). Every lane is
    padded by cycling back to its own first scene (whose first frame carries
    ``prev_exists == 0``, so the model resets its memory as at any scene start)
    or truncated so that every rank serves exactly
    ``ceil(total_frames / (world_size * batch_size))`` rounds. A constant
    ``len()`` is what Lightning assumes: it derives ``val_check_batch`` and
    ``estimated_stepping_batches`` once from the first epoch, and a shorter
    later epoch would silently skip its validation. Padding follows the
    ``DistributedSampler`` convention of repeating a few indices instead of
    dropping frames.

    Without ``shuffle`` (evaluation) the scene order is deterministic, so the
    length is constant anyway; lanes are trimmed to the shortest one and no
    frame is ever scored twice. Evaluation should run one lane per rank so
    that trimming drops nothing.

    The sampler shards by rank itself, so the trainer must run with
    ``use_distributed_sampler: false`` (the StreamPETR base config does):
    Lightning would otherwise wrap it in a ``DistributedSamplerWrapper``,
    re-striding the indices and destroying the lane/scene contiguity.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """Initialize the streaming sampler.

        Args:
            dataset: Dataset exposing ``scene_index_groups()`` with
                scene-contiguous dataset indices.
            batch_size: Number of dataloader lanes fed in parallel. Must match
                the dataloader batch size.
            shuffle: Whether to shuffle the scene order every epoch. Also
                selects the fixed-length (padded) epoch described above.
            seed: Base seed for the per-epoch scene shuffle.
        """
        self.scene_groups = dataset.scene_index_groups()
        if not self.scene_groups:
            raise ValueError("GroupStreamingSampler requires at least one scene group.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_indices: list[list[int]] = []
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.total_frames = sum(len(group) for group in self.scene_groups)
        self.rounds_per_epoch = math.ceil(self.total_frames / (self.world_size * self.batch_size))

    def _scene_order(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.scene_groups)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(len(self.scene_groups), generator=generator).tolist()

    def _rank_lanes(self, scene_order: list[int], rank: int) -> list[list[int]]:
        """Assign this rank's scenes to lanes, each scene to the shortest lane."""
        lanes: list[list[int]] = [[] for _ in range(self.batch_size)]
        for scene_index in scene_order[rank :: self.world_size]:
            min(lanes, key=len).extend(self.scene_groups[scene_index])
        return lanes

    @staticmethod
    def _fit_lane(lane: list[int], rounds: int) -> list[int]:
        """Cycle a lane back to its own start, or cut it, to exactly ``rounds`` frames."""
        if not lane:
            return lane
        repeats = math.ceil(rounds / len(lane))
        return (lane * repeats)[:rounds]

    def _epoch_indices(self, epoch: int) -> list[list[int]]:
        """Build (or reuse) the per-rank interleaved index lists for one epoch."""
        if epoch == self._cached_epoch:
            return self._cached_indices
        scene_order = self._scene_order(epoch)

        per_rank_indices: list[list[int]] = []
        for rank in range(self.world_size):
            lanes = self._rank_lanes(scene_order, rank)
            if self.shuffle:
                lanes = [self._fit_lane(lane, self.rounds_per_epoch) for lane in lanes]
            rounds = min(len(lane) for lane in lanes)
            indices = [lane[round_index] for round_index in range(rounds) for lane in lanes]
            per_rank_indices.append(indices)

        min_length = min(len(indices) for indices in per_rank_indices)
        if min_length == 0:
            raise ValueError(
                f"GroupStreamingSampler cannot fill {self.world_size} rank(s) x "
                f"{self.batch_size} lane(s) from {len(self.scene_groups)} scene(s); "
                "reduce batch_size or world size."
            )
        self._cached_epoch = epoch
        self._cached_indices = [indices[:min_length] for indices in per_rank_indices]
        served = min_length * self.world_size
        if self.shuffle:
            distinct = len({index for indices in self._cached_indices for index in indices})
            logger.info(
                "GroupStreamingSampler epoch %d: %d/%d distinct frames, %d repeated to keep "
                "%d rank(s) x %d lane(s) at a fixed %d rounds.",
                epoch,
                distinct,
                self.total_frames,
                served - distinct,
                self.world_size,
                self.batch_size,
                self.rounds_per_epoch,
            )
        elif served < self.total_frames:
            logger.warning(
                "GroupStreamingSampler serves %d/%d frames; %d tail frames are trimmed to keep "
                "%d rank(s) x %d lane(s) aligned. Evaluate with one lane per rank to score "
                "every frame.",
                served,
                self.total_frames,
                self.total_frames - served,
                self.world_size,
                self.batch_size,
            )
        return self._cached_indices

    def __iter__(self):
        """Yield this rank's interleaved indices for the current epoch."""
        return iter(self._epoch_indices(self.epoch)[self.rank])

    def __len__(self) -> int:
        """Return the per-rank sample count; constant across epochs when shuffling."""
        return len(self._epoch_indices(self.epoch)[self.rank])

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used for the scene shuffle.

        Lightning calls this before every training epoch, which is what
        advances the shuffle; without it every epoch replays the same order.
        """
        self.epoch = epoch
