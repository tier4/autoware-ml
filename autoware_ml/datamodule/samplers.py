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

"""Repeat factor sampling of the training frames.

A detection corpus is dominated by cars, and a frame that carries a rare class is drawn as
often as one that does not. Repeat factor sampling gives every frame a weight from the
rarest category it carries, so frames with rare objects come up more often within an epoch
of the same length. The weights come straight from the record table, before any transform.
"""

from __future__ import annotations

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
    """
    Draw one weighted epoch and hand every rank its share of it.

    An epoch has as many draws as the dataset has samples, drawn with replacement in
    proportion to the weights. Every rank draws the same epoch from the same seed and keeps
    every world-size-th draw. Lightning leaves a distributed sampler in place and sets its
    epoch, which reseeds the draw.
    """

    def __init__(self, dataset: Dataset, weights: Sequence[float], seed: int) -> None:
        """
        Initialize the sampler.

        Args:
          dataset: Dataset the loader indexes.
          weights: Sampling weight of every sample, all positive.
          seed: Seed of the first epoch, every epoch adds its number.
        """
        if len(weights) != len(dataset):
            raise ValueError(f"Got {len(weights)} weights for {len(dataset)} samples.")
        distributed = dist.is_available() and dist.is_initialized()
        super().__init__(
            dataset,
            num_replicas=dist.get_world_size() if distributed else 1,
            rank=dist.get_rank() if distributed else 0,
            shuffle=False,
            seed=seed,
        )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if torch.any(self.weights <= 0.0):
            raise ValueError("Sampling weights must be positive.")

    def __iter__(self):
        """Yield the indices of this rank for the current epoch."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])
