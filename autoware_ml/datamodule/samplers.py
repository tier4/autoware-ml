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

"""Data samplers shared by Autoware-ML datamodules.

A detection corpus is dominated by cars, and a frame that carries a rare class is drawn as
often as one that does not. Repeat factor sampling gives every frame a weight from the
rarest category it carries, so frames with rare objects come up more often within an epoch
of the same length. The weights come straight from the record table, before any transform.
The scene streaming sampler feeds temporal models whole scenes frame by frame.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.base_dataset import BaseDataset, ConcatDataset
from autoware_ml.transforms.boxes3d.filters import parse_exclusion_rules
from autoware_ml.types.geometry import Box3DFieldIndex

logger = logging.getLogger(__name__)

LOW_PEDESTRIAN_CATEGORY = "low_pedestrian"


@dataclass(frozen=True)
class FrameSamplingConfig:
    """
    Settings of the repeat factor sampling.

    Attributes:
      repeat_sampling_factor: Category fraction below which a category is oversampled. The
        repeat factor of a category is the square root of this over its fraction, at least one.
      object_bev_range: BEV range [x_min, y_min, x_max, y_max] a box center has to fall in for
        the box to be counted.
      pedestrian_class_name: Class whose short boxes count as low pedestrians.
      low_pedestrian_height_threshold: Height in meters below which a pedestrian counts as a
        low pedestrian, for example a child or a person bending down.
      low_pedestrian_bev_range: BEV range the center of a low pedestrian has to fall in.
      class_names: Trained class names, the categories a box is counted into.
      ignore_index: Label index of the boxes of no trained class.
      filter_attributes: Class and attribute pairs of boxes that are not detection targets,
        same as the attribute filter transform.
      seed: Seed of the draw, every epoch adds its number.
    """

    repeat_sampling_factor: float
    object_bev_range: Sequence[float]
    pedestrian_class_name: str
    low_pedestrian_height_threshold: float
    low_pedestrian_bev_range: Sequence[float]
    class_names: Sequence[str]
    ignore_index: int
    filter_attributes: Sequence[Sequence[str]]
    seed: int

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not 0.0 < self.repeat_sampling_factor <= 1.0:
            raise ValueError(
                f"repeat_sampling_factor must be in (0, 1], got {self.repeat_sampling_factor}."
            )
        for name in ("object_bev_range", "low_pedestrian_bev_range"):
            bev_range = getattr(self, name)
            if len(bev_range) != 4 or bev_range[0] >= bev_range[2] or bev_range[1] >= bev_range[3]:
                raise ValueError(
                    f"{name} must be [x_min, y_min, x_max, y_max] with min below max, got "
                    f"{bev_range}."
                )
        if self.pedestrian_class_name not in self.class_names:
            raise ValueError(
                f"The pedestrian class {self.pedestrian_class_name!r} is not a trained class."
            )
        if LOW_PEDESTRIAN_CATEGORY in self.class_names:
            raise ValueError(
                f"The low pedestrian category {LOW_PEDESTRIAN_CATEGORY!r} clashes with a "
                "trained class."
            )
        parse_exclusion_rules(self.filter_attributes)

    @property
    def excluded(self) -> frozenset[tuple[str, str]]:
        """Class and attribute pairs of the boxes that are not counted."""
        return parse_exclusion_rules(self.filter_attributes)

    @property
    def categories(self) -> tuple[str, ...]:
        """Sampling categories, the trained classes and the low pedestrian bucket."""
        return (*self.class_names, LOW_PEDESTRIAN_CATEGORY)


def compute_frame_sampling_weights(
    dataset: ConcatDataset, config: FrameSamplingConfig
) -> list[float]:
    """
    Weight of every sample of a training split.

    Each category gets a repeat factor from its share of the frames and of the boxes. A sample's
    weight is the largest factor of its categories. Samples from sources whose boxes are not
    used have weight 1.

    Args:
      dataset: Concatenated training split, repeats included.
      config: Repeat factor settings.

    Returns:
      list[float]: One weight per sample index of the split.
    """
    source_samples = [
        _source_sample_categories(source, config) if source.det3d_supervised else None
        for source in dataset.datasets
    ]

    num_supervised_samples = 0
    frame_counts = dict.fromkeys(config.categories, 0)
    box_counts = dict.fromkeys(config.categories, 0)
    for samples, repeat in zip(source_samples, dataset.repeats, strict=True):
        if samples is None:
            continue
        num_supervised_samples += len(samples) * repeat
        for categories in samples:
            for category, count in categories.items():
                frame_counts[category] += repeat
                box_counts[category] += count * repeat
    total_boxes = sum(box_counts.values())
    if total_boxes == 0:
        raise ValueError("Repeat factor sampling needs a training split with at least one box.")

    factors = {}
    for category in config.categories:
        if frame_counts[category] == 0:
            factors[category] = 1.0
            continue
        frame_fraction = frame_counts[category] / num_supervised_samples
        box_fraction = box_counts[category] / total_boxes
        category_fraction = math.sqrt(frame_fraction * box_fraction)
        factors[category] = max(1.0, math.sqrt(config.repeat_sampling_factor / category_fraction))

    weights: list[float] = []
    for source, samples, repeat in zip(
        dataset.datasets, source_samples, dataset.repeats, strict=True
    ):
        if samples is None:
            weights.extend([1.0] * (len(source) * repeat))
            continue
        source_weights = [
            max((factors[category] for category in categories), default=1.0)
            for categories in samples
        ]
        weights.extend(source_weights * repeat)
    return weights


def _source_sample_categories(
    source: BaseDataset, config: FrameSamplingConfig
) -> list[dict[str, int]]:
    """
    Category counts of every sample of one source.

    Args:
      source: Source dataset carrying the record table.
      config: Repeat factor settings.

    Returns:
      list[dict[str, int]]: Per sample, the number of counted boxes of every category present.
    """
    boxes_column = source.dataset_records_dataframe.get_column(DatasetTableSchema.BOXES_3D.name)
    record_counts = [_record_category_counts(boxes, config) for boxes in boxes_column.to_list()]
    return [record_counts[source.record_index(index)] for index in range(len(source))]


def _record_category_counts(
    boxes: Sequence[Mapping[str, Any]], config: FrameSamplingConfig
) -> dict[str, int]:
    """
    Category counts of one record.

    A box counts when it is valid, of a trained class, not excluded by an attribute rule,
    seen by the lidar and centered inside the object range. A short pedestrian close to the
    vehicle counts as a low pedestrian instead of a pedestrian.

    Args:
      boxes: Boxes of the record as the record table stores them.
      config: Repeat factor settings.

    Returns:
      dict[str, int]: Number of counted boxes per category present in the record.
    """
    excluded = config.excluded
    counts: dict[str, int] = {}
    for box in boxes:
        # A box keeps its annotated name, its class is read from its label index
        label_index = box[Box3DDatasetSchema.BOX3D_LABEL_INDEX.name]
        if label_index == config.ignore_index:
            continue
        if not 0 <= label_index < len(config.class_names):
            raise ValueError(
                f"Box label index {label_index} is outside the {len(config.class_names)} "
                "trained classes."
            )
        if not box[Box3DDatasetSchema.BOX3D_VALID.name]:
            continue
        class_name = config.class_names[label_index]
        label_name = box[Box3DDatasetSchema.BOX3D_LABEL_NAME.name]
        if box[Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTS.name] <= 0:
            continue
        attributes = box[Box3DDatasetSchema.BOX3D_ATTRIBUTES.name]
        if any((label_name, attribute) in excluded for attribute in attributes):
            continue
        params = box[Box3DDatasetSchema.BOX3D_PARAMS.name]
        if not _center_in_range(params, config.object_bev_range):
            continue
        category = class_name
        if (
            class_name == config.pedestrian_class_name
            and params[Box3DFieldIndex.HEIGHT] < config.low_pedestrian_height_threshold
            and _center_in_range(params, config.low_pedestrian_bev_range)
        ):
            category = LOW_PEDESTRIAN_CATEGORY
        counts[category] = counts.get(category, 0) + 1
    return counts


def _center_in_range(params: Sequence[float], bev_range: Sequence[float]) -> bool:
    """Whether the box center lies inside [x_min, y_min, x_max, y_max]."""
    x, y = params[Box3DFieldIndex.X], params[Box3DFieldIndex.Y]
    return bev_range[0] <= x <= bev_range[2] and bev_range[1] <= y <= bev_range[3]


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


class GroupStreamingSampler(DistributedSampler):
    """Feed every batch position the frames of one scene after another, in time order.

    A lane is one position of a rank's batch. Across iterations a lane walks its frames
    one after another, so a temporal model can keep one memory per lane; it tells from
    the frame metadata of the batch where a lane starts a new stream and resets that
    lane's memory there.

    Every epoch the scenes are joined, in order or shuffled by ``seed + epoch``, and their
    frames are cut into ``num_lanes`` (``num_replicas * batch_size``) lanes whose lengths
    differ by at most one; rank ``r`` serves lanes ``r * batch_size`` to
    ``(r + 1) * batch_size - 1``. A cut can fall inside a scene, and the lane after it
    starts a new stream there.

    Every rank serves the same number of rounds, padded or trimmed as below, so no rank
    waits forever in a DDP collective (the gradient sync of every training step, or an
    ``all_reduce`` a model may run in its validation loss).

    - ``shuffle=True`` (training): every lane is padded to
      ``ceil(total_frames / num_lanes)`` rounds by restarting at its own first frame. No
      frame is dropped and fewer than ``num_lanes`` repeat. The epoch length never
      changes: Lightning derives the validation interval and the total step count once,
      from the first epoch, and a shorter later epoch would silently skip its validation.
    - ``shuffle=False`` (evaluation): every lane is trimmed to
      ``floor(total_frames / num_lanes)`` rounds. No frame is scored twice and fewer than
      ``num_lanes`` tail frames go unscored, none with one rank and one lane. A prediction
      has to serve every frame, so the datamodule refuses one that would trim a frame.

    As a ``DistributedSampler`` it is never wrapped by Lightning, whose wrapper would keep
    only every ``num_replicas``-th index of each rank and make its lanes skip frames.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        """Initialize the streaming sampler.

        Args:
            dataset: Dataset exposing ``scene_index_groups()``, the indices of
                every scene in frame order.
            batch_size: Number of dataloader lanes fed in parallel. Must match
                the dataloader batch size.
            shuffle: Whether to shuffle the scene order every epoch. Also
                selects the fixed-length (padded) epoch described above.
            seed: Base seed for the per-epoch scene shuffle, the experiment seed.
        """
        num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.scene_groups = dataset.scene_index_groups()
        if not self.scene_groups:
            raise ValueError("GroupStreamingSampler requires at least one scene group.")
        self.batch_size = batch_size
        self._cached_epoch: int | None = None
        self._cached_indices: list[list[int]] = []
        self._trimming_warned = False
        self.total_frames = sum(len(group) for group in self.scene_groups)

    @property
    def num_lanes(self) -> int:
        """Lanes over every rank, ``num_replicas * batch_size``."""
        return self.num_replicas * self.batch_size

    @property
    def rounds_per_epoch(self) -> int:
        """Rounds every rank serves per shuffled epoch, the ``DistributedSampler`` convention."""
        return math.ceil(self.total_frames / self.num_lanes)

    def _scene_order(self, epoch: int) -> list[int]:
        """Order the scenes for one epoch: as listed, or shuffled by the epoch's seed."""
        if not self.shuffle:
            return list(range(len(self.scene_groups)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(len(self.scene_groups), generator=generator).tolist()

    def _lanes(self, scene_order: list[int]) -> list[list[int]]:
        """Cut the joined frames of the epoch into ``num_lanes`` lanes of (almost) equal length."""
        frames = [index for scene_index in scene_order for index in self.scene_groups[scene_index]]
        bounds = [lane * len(frames) // self.num_lanes for lane in range(self.num_lanes + 1)]
        return [frames[start:end] for start, end in zip(bounds[:-1], bounds[1:])]

    @staticmethod
    def _fit_lane(lane: list[int], rounds: int) -> list[int]:
        """Cycle a lane back to its own start, or cut it, to exactly ``rounds`` frames."""
        repeats = math.ceil(rounds / len(lane))
        return (lane * repeats)[:rounds]

    def _epoch_indices(self, epoch: int) -> list[list[int]]:
        """Build (or reuse) the per-rank interleaved index lists for one epoch."""
        if epoch == self._cached_epoch:
            return self._cached_indices
        lanes = self._lanes(self._scene_order(epoch))
        if any(not lane for lane in lanes):
            raise ValueError(
                f"GroupStreamingSampler cannot fill {self.num_replicas} rank(s) x "
                f"{self.batch_size} lane(s) from {self.total_frames} frame(s); "
                "reduce batch_size or world size."
            )
        rounds = self.rounds_per_epoch if self.shuffle else self.total_frames // self.num_lanes
        lanes = [self._fit_lane(lane, rounds) for lane in lanes]

        self._cached_epoch = epoch
        self._cached_indices = [
            [lane[round_index] for round_index in range(rounds) for lane in rank_lanes]
            for rank_lanes in (
                lanes[rank * self.batch_size : (rank + 1) * self.batch_size]
                for rank in range(self.num_replicas)
            )
        ]
        served = rounds * self.num_lanes
        if self.shuffle:
            logger.info(
                "GroupStreamingSampler epoch %d: %d frames, %d repeated to keep %d rank(s) x "
                "%d lane(s) at a fixed %d rounds.",
                epoch,
                self.total_frames,
                served - self.total_frames,
                self.num_replicas,
                self.batch_size,
                rounds,
            )
        elif served < self.total_frames and not self._trimming_warned:
            # Lightning sets the epoch on the evaluation samplers too, which rebuilds the
            # indices at every validation; the frames trimmed never change, so say it once.
            self._trimming_warned = True
            logger.warning(
                "GroupStreamingSampler serves %d/%d frames; %d tail frames are trimmed to keep "
                "%d rank(s) x %d lane(s) aligned. Only one rank with one lane scores every "
                "frame.",
                served,
                self.total_frames,
                self.total_frames - served,
                self.num_replicas,
                self.batch_size,
            )
        return self._cached_indices

    def __iter__(self):
        """Yield this rank's interleaved indices for the current epoch."""
        return iter(self._epoch_indices(self.epoch)[self.rank])

    def __len__(self) -> int:
        """Return the per-rank sample count; constant across epochs when shuffling."""
        return len(self._epoch_indices(self.epoch)[self.rank])
