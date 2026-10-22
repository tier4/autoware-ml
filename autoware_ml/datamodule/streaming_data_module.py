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

"""Datamodule streaming the frames of every scene in time order, for temporal models."""

from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import DataLoader

from autoware_ml.datamodule.data_module import DataLoaderConfig, DataModule
from autoware_ml.datamodule.samplers import FrameSamplingConfig, GroupStreamingSampler
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType


class StreamingDataModule(DataModule):
    """DataModule feeding every batch position the frames of one scene after another.

    A temporal model carries a memory across batches, one per batch position, so every
    dataloader draws from a ``GroupStreamingSampler``, which lays out the lanes and
    reshuffles the scenes of the training split every epoch from ``seed``. The samples it adds
    only to pad a lane are flagged in ``ModelGTBatch.padding_mask``.

    Whatever else would reorder the frames is refused: frame sampling, a shuffling dataloader
    (which ``DataLoader`` itself refuses next to a sampler) and ``trainer.overfit_batches``, for
    which Lightning swaps the sampler of the training and validation loaders for a sequential
    one.
    """

    def __init__(
        self,
        splitter: SplitterInterface,
        sources: Mapping[str, DatasetSource],
        train_dataloader: DataLoaderConfig | None,
        validation_dataloader: DataLoaderConfig | None,
        test_dataloader: DataLoaderConfig | None,
        predict_dataloader: DataLoaderConfig | None,
        train_frame_sampling: FrameSamplingConfig | None,
        seed: int,
    ) -> None:
        """
        Initialize the datamodule.

        Args:
          splitter: Splitter assigning the records of a database to its splits.
          sources: Dataset sources by name, in the order their samples are concatenated.
          train_dataloader: Dataloader settings of the training split.
          validation_dataloader: Dataloader settings of the validation split.
          test_dataloader: Dataloader settings of the test split.
          predict_dataloader: Dataloader settings of the predict split.
          train_frame_sampling: Must be None. Taken so that the datamodule composes from the
            same config as ``DataModule``.
          seed: Seed of the per-epoch scene shuffle of the training split; pass the
            experiment seed so every run draws its own order.

        Raises:
          ValueError: If frame sampling is set, which would draw the frames of a scene apart.
        """
        if train_frame_sampling is not None:
            raise ValueError(
                "Scene streaming serves the frames of every scene in time order, so it cannot "
                "be combined with frame sampling. Set train_frame_sampling to null."
            )
        super().__init__(
            splitter=splitter,
            sources=sources,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            test_dataloader=test_dataloader,
            predict_dataloader=predict_dataloader,
            train_frame_sampling=None,
        )
        self.seed = seed

    def build_dataloader(self, split: SplitType, config: DataLoaderConfig | None) -> DataLoader:
        """
        Build the streaming dataloader of one split.

        Args:
          split: Split the dataloader serves.
          config: Dataloader settings of the split, whose batch size is the number of lanes
            of every rank.

        Returns:
          DataLoader: Dataloader over the concatenated sources of the split, drawing from a
            ``GroupStreamingSampler`` that reshuffles the scenes of the training split only.

        Raises:
          ValueError: If the split has no dataset or no dataloader settings, if the dataloader
            shuffles, or if Lightning would swap the sampler for ``trainer.overfit_batches``.
        """
        dataset = self.datasets.get(split)
        if dataset is None:
            raise ValueError(
                f"Split {split} has no dataset. Declare a source serving it and call setup()."
            )
        if config is None:
            raise ValueError(f"Split {split} has no dataloader settings.")
        self._validate_frame_order(split)

        sampler = GroupStreamingSampler(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == SplitType.TRAIN,
            seed=self.seed,
        )
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            **config.to_dataloader_kwargs(),
        )

    def _validate_frame_order(self, split: SplitType) -> None:
        """
        Reject a dataloader whose frames would not reach the lanes in the sampler's order.

        Args:
          split: Split the dataloader serves.

        Raises:
          ValueError: If a training or validation loader serves a trainer with
            ``overfit_batches``, for which Lightning swaps the scene sampler for a sequential
            one.
        """
        if (
            split in (SplitType.TRAIN, SplitType.VAL)
            and self.trainer is not None
            and self.trainer.overfit_batches > 0
        ):
            raise ValueError(
                "trainer.overfit_batches makes Lightning swap the scene sampler of the "
                "training and validation loaders for a sequential one, which feeds every "
                "lane each batch_size-th sample as if it were one stream. Shorten a "
                "streaming run with limit_train_batches / limit_val_batches instead."
            )
