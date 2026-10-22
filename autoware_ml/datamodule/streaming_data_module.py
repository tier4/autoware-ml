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
from typing import ClassVar

from torch.utils.data import DataLoader

from autoware_ml.datamodule.data_module import DataLoaderConfig, DataModule
from autoware_ml.datamodule.samplers import FrameSamplingConfig, GroupStreamingSampler
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType


class StreamingDataModule(DataModule):
    """DataModule feeding every batch position the frames of one scene after another.

    A temporal model carries a memory across batches, one per batch position, so every
    dataloader draws from a ``GroupStreamingSampler``. Every epoch the sampler joins the
    scenes, the training split reshuffling their order from ``seed``, and cuts their frames,
    each scene in time order, into one lane per batch position of every rank; a cut can fall
    inside a scene, and the lane after it starts a new stream there. The sampler is a
    ``DistributedSampler``, so it splits the lanes over the ranks itself and Lightning leaves
    it alone.

    Whatever else would reorder the frames is refused: frame sampling, a shuffling dataloader
    and ``trainer.overfit_batches``, for which Lightning swaps the sampler of the training and
    validation loaders for a sequential one. To keep the lanes aligned, an evaluation loader
    leaves out the last frame of the longer lanes, fewer frames than there are lanes, so a
    prediction that would leave a frame out is refused.
    """

    # A model carrying a memory across batches reads it from the datamodule of its trainer.
    streaming: ClassVar[bool] = True

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
          ValueError: If the split has no dataset or no dataloader settings, if the
            dataloader would reorder the frames, or if a prediction would leave frames out.
        """
        dataset = self.datasets.get(split)
        if dataset is None:
            raise ValueError(
                f"Split {split} has no dataset. Declare a source serving it and call setup()."
            )
        if config is None:
            raise ValueError(f"Split {split} has no dataloader settings.")
        self._validate_frame_order(split, config)

        sampler = GroupStreamingSampler(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == SplitType.TRAIN,
            seed=self.seed,
        )
        if split == SplitType.PREDICT:
            self._validate_every_frame_is_predicted(sampler)
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            **config.to_dataloader_kwargs(),
        )

    def _validate_frame_order(self, split: SplitType, config: DataLoaderConfig) -> None:
        """
        Reject a dataloader whose frames would not reach the lanes in the sampler's order.

        Args:
          split: Split the dataloader serves.
          config: Dataloader settings of the split.

        Raises:
          ValueError: If the dataloader shuffles, or if a training or validation loader
            serves a trainer with ``overfit_batches``, for which Lightning swaps the scene
            sampler for a sequential one.
        """
        if config.shuffle:
            raise ValueError(
                "Scene streaming owns the sample order, so the dataloader cannot shuffle. "
                "Set shuffle to false."
            )
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

    @staticmethod
    def _validate_every_frame_is_predicted(sampler: GroupStreamingSampler) -> None:
        """
        Reject a prediction whose lanes would leave frames out.

        Like evaluation, prediction trims the longer lanes to keep them aligned. A frame it
        left out could pass for a frame without objects, in pseudo labels for example.

        Args:
          sampler: Scene sampler of the predict split.

        Raises:
          ValueError: If the frames of the split do not fill every lane equally.
        """
        skipped = sampler.total_frames % sampler.num_lanes
        if skipped:
            raise ValueError(
                f"Streaming prediction over {sampler.num_lanes} lane(s) would skip "
                f"{skipped} of {sampler.total_frames} frames to keep the lanes aligned. "
                "Predict on one device with batch_size 1 to serve every frame."
            )
