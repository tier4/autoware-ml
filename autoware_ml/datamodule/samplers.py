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
    dataset's ``prev_exists`` metadata. Scenes are shuffled per epoch,
    partitioned round-robin over distributed ranks and lanes, and every rank
    is trimmed to the shortest rank so all batches stay complete.
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
            shuffle: Whether to shuffle the scene order every epoch.
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

    def _epoch_indices(self, epoch: int) -> list[list[int]]:
        """Build (or reuse) the per-rank interleaved index lists for one epoch."""
        if epoch == self._cached_epoch:
            return self._cached_indices
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch)
            scene_order = torch.randperm(len(self.scene_groups), generator=generator).tolist()
        else:
            scene_order = list(range(len(self.scene_groups)))

        per_rank_indices: list[list[int]] = []
        for rank in range(self.world_size):
            lanes: list[list[int]] = [[] for _ in range(self.batch_size)]
            for scene_position, scene_index in enumerate(scene_order[rank :: self.world_size]):
                lanes[scene_position % self.batch_size].extend(self.scene_groups[scene_index])
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
        total_frames = sum(len(group) for group in self.scene_groups)
        dropped = total_frames - min_length * self.world_size
        if dropped:
            logger.info(
                "GroupStreamingSampler serves %d/%d frames this epoch; %d tail frames are "
                "trimmed to keep %d rank(s) x %d lane(s) aligned.",
                min_length * self.world_size,
                total_frames,
                dropped,
                self.world_size,
                self.batch_size,
            )
        return self._cached_indices

    def __iter__(self):
        """Yield this rank's interleaved indices for the current epoch."""
        return iter(self._epoch_indices(self.epoch)[self.rank])

    def __len__(self) -> int:
        """Return the per-rank sample count of the current epoch."""
        return len(self._epoch_indices(self.epoch)[self.rank])

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used for the scene shuffle.

        Lightning calls this before every training epoch, which is what
        advances the shuffle; without it every epoch replays the same order.
        """
        self.epoch = epoch
