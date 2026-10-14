import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L
from torch.utils.data import DataLoader

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.datamodule.base_dataset import ConcatDataset
from autoware_ml.datamodule.samplers import (
    DistributedWeightedRandomSampler,
    FrameSamplingConfig,
    compute_frame_sampling_weights,
)
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)

# Split of the scenario lists every datamodule split reads its records from. Prediction runs
# on the scenarios of the test split.
_RECORD_SPLIT = {
    SplitType.TRAIN: SplitType.TRAIN,
    SplitType.VAL: SplitType.VAL,
    SplitType.TEST: SplitType.TEST,
    SplitType.PREDICT: SplitType.TEST,
}


@dataclass
class DataLoaderConfig:
    """Store configuration values for one dataloader.

    Attributes:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used by the dataloader.
        pin_memory: Whether to pin host memory before device transfer.
        persistent_workers: Whether worker processes stay alive across epochs.
        shuffle: Whether the dataloader shuffles samples.
        drop_last: Whether to drop the final incomplete batch.
    """

    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
    drop_last: bool = False

    def to_dataloader_kwargs(self) -> dict[str, Any]:
        """Convert to keyword arguments accepted by ``DataLoader``.

        Returns:
            Dictionary of DataLoader constructor keyword arguments.
        """
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
            "drop_last": self.drop_last,
        }


class DataModule(L.LightningDataModule):
    """LightningDataModule shared by every task and database.

    Every dataset source names a database and the dataset of each split it serves. A split is
    served by the sources declaring a dataset for it, concatenated in declaration order, and
    the repeat of a source sets its share of the training epoch. Preparation builds the record
    table of each database once. Setup splits each table by its scenario lists and binds the
    dataset of every source to the records of its database.
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
          train_frame_sampling: Repeat factor sampling of the training split, None when its
            samples are drawn uniformly.
        """
        super().__init__()

        self.splitter = splitter
        if not len(sources):
            raise ValueError("The datamodule needs at least one dataset source.")

        self.sources: dict[str, DatasetSource] = dict(sources)
        self._validate_shared_taxonomy()
        self.datasets: dict[SplitType, ConcatDataset] = {}
        self.train_dataloader_config = train_dataloader
        self.validation_dataloader_config = validation_dataloader
        self.test_dataloader_config = test_dataloader
        self.predict_dataloader_config = predict_dataloader
        self.train_frame_sampling = train_frame_sampling

    def _validate_shared_taxonomy(self) -> None:
        """
        Reject sources whose taxonomies disagree.

        Sources mixed in one run must use the same class indices, otherwise one output would
        learn two classes.
        """
        taxonomies = {
            source.database.taxonomy: source.database.version for source in self.sources.values()
        }
        if len(taxonomies) > 1:
            raise ValueError(
                "All dataset sources must use the same taxonomy, got different ones in "
                f"{sorted(taxonomies.values())}."
            )

    def databases(self) -> Sequence[BaseDatabase]:
        """
        Every distinct database of every source, in declaration order.

        Returns:
          Sequence[BaseDatabase]: Databases the datamodule reads, one per database hash.
        """
        databases: dict[str, BaseDatabase] = {}
        for source in self.sources.values():
            databases.setdefault(source.database.database_hash, source.database)
        return tuple(databases.values())

    def split_sources(self, split: SplitType) -> Mapping[str, DatasetSource]:
        """
        Sources serving one split, in declaration order.

        Args:
          split: Split asked for.

        Returns:
          Mapping[str, DatasetSource]: The sources declaring a dataset for the split, by name.
        """
        return {
            name: source
            for name, source in self.sources.items()
            if source.dataset(split) is not None
        }

    def setup(self, stage: str | None = None) -> None:
        """Build the dataset of every split the stage needs.

        Each database of a split is read and split once. Every source of the split binds a copy
        of its dataset to the records of its database, and the bound datasets are concatenated
        in declaration order.

        Args:
            stage: Lightning stage, one of None, fit, validate, test or predict.
        """
        stage_to_splits = {
            None: (SplitType.TRAIN, SplitType.VAL, SplitType.TEST, SplitType.PREDICT),
            "fit": (SplitType.TRAIN, SplitType.VAL),
            "validate": (SplitType.VAL,),
            "test": (SplitType.TEST,),
            "predict": (SplitType.PREDICT,),
        }

        split_records: dict[str, Mapping[SplitType, Any]] = {}
        for split in stage_to_splits[stage]:
            sources = self.split_sources(split)
            if not sources:
                logger.info(f"No dataset source serves split {split}, skipping it.")
                continue

            datasets = []
            repeats = []
            for name, source in sources.items():
                database = source.database
                if database.database_hash not in split_records:
                    logger.info(f"Splitting the records of database {database.version}...")
                    split_records[database.database_hash] = self.splitter.split_by_polars_dataframe(
                        dataset_records_dataframe=database.load_polars_scenario_dataframe(),
                        scenarios=database.scenarios,
                    )
                records = split_records[database.database_hash][_RECORD_SPLIT[split]]
                repeat = source.repeat if split == SplitType.TRAIN else 1
                dataset = source.dataset(split)
                assert dataset is not None
                bound = dataset.bind(records)
                logger.info(
                    f"Source {name} serves {len(records)} records of database "
                    f"{database.version} to split {split}, det3d {bound.det3d_supervised}, "
                    f"seg3d {bound.seg3d_supervised}, repeated {repeat} times."
                )
                datasets.append(bound)
                repeats.append(repeat)

            self.datasets[split] = ConcatDataset(datasets=datasets, repeats=repeats)
            logger.info(f"Split {split} serves {len(self.datasets[split])} samples.")

    def prepare_data(self) -> None:
        """
        Build the record table of every database that has no cache yet.

        Lightning calls this in a single process of the main node.
        """
        logger.info("Preparing the record tables of every database...")
        for database in self.databases():
            database.process_scenario_records()
        logger.info("Finished preparing the record tables.")

    def build_dataloader(self, split: SplitType, config: DataLoaderConfig | None) -> DataLoader:
        """
        Build the dataloader of one split.

        Args:
          split: Split the dataloader serves.
          config: Dataloader settings of the split.

        Returns:
          DataLoader: Dataloader over the concatenated sources of the split.

        Raises:
          ValueError: If the split has no dataset or no dataloader settings, or if frame
            sampling is on while the training dataloader shuffles.
        """
        dataset = self.datasets.get(split)
        if dataset is None:
            raise ValueError(
                f"Split {split} has no dataset. Declare a source serving it and call setup()."
            )
        if config is None:
            raise ValueError(f"Split {split} has no dataloader settings.")

        kwargs = config.to_dataloader_kwargs()
        if split == SplitType.TRAIN and self.train_frame_sampling is not None:
            if config.shuffle:
                raise ValueError(
                    "The training dataloader cannot shuffle when frame sampling is on, the "
                    "weighted sampler draws the order. Set shuffle to false."
                )
            weights = compute_frame_sampling_weights(dataset, self.train_frame_sampling)
            kwargs["sampler"] = DistributedWeightedRandomSampler(
                dataset,
                weights,
                seed=self.train_frame_sampling.seed,
                drop_last=config.drop_last,
            )
        return DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Create the dataloader of the training split."""
        return self.build_dataloader(SplitType.TRAIN, self.train_dataloader_config)

    def val_dataloader(self) -> DataLoader:
        """Create the dataloader of the validation split."""
        return self.build_dataloader(SplitType.VAL, self.validation_dataloader_config)

    def test_dataloader(self) -> DataLoader:
        """Create the dataloader of the test split."""
        return self.build_dataloader(SplitType.TEST, self.test_dataloader_config)

    def predict_dataloader(self) -> DataLoader:
        """Create the dataloader of the predict split."""
        return self.build_dataloader(SplitType.PREDICT, self.predict_dataloader_config)
