import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
from torch.utils.data import DataLoader

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.datamodule.base import DataLoaderConfig
from autoware_ml.datamodule.base_dataset import BaseDataset, ConcatDataset
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


class DataModule(L.LightningDataModule):
    """LightningDataModule shared by every task and database.

    Every split lists the dataset sources it reads. Each source keeps its own database, root
    directory and used annotations, and its repeat sets its share of the epoch. Preparation
    builds the record table of each database once. Setup splits each table by its scenario
    lists and builds one dataset per source.
    """

    def __init__(
        self,
        splitter: SplitterInterface,
        train_sources: Sequence[Mapping[str, Any]] | None,
        validation_sources: Sequence[Mapping[str, Any]] | None,
        test_sources: Sequence[Mapping[str, Any]] | None,
        predict_sources: Sequence[Mapping[str, Any]] | None,
        train_dataset: Callable[..., BaseDataset] | None,
        validation_dataset: Callable[..., BaseDataset] | None,
        test_dataset: Callable[..., BaseDataset] | None,
        predict_dataset: Callable[..., BaseDataset] | None,
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
          train_sources: Dataset sources of the training split, each one the fields of a
            DatasetSource. None when the split is unused.
          validation_sources: Dataset sources of the validation split.
          test_sources: Dataset sources of the test split.
          predict_sources: Dataset sources of the predict split.
          train_dataset: Dataset factory of the training split, called once per source.
          validation_dataset: Dataset factory of the validation split.
          test_dataset: Dataset factory of the test split.
          predict_dataset: Dataset factory of the predict split.
          train_dataloader: Dataloader settings of the training split.
          validation_dataloader: Dataloader settings of the validation split.
          test_dataloader: Dataloader settings of the test split.
          predict_dataloader: Dataloader settings of the predict split.
          train_frame_sampling: Repeat factor sampling of the training split, None when its
            samples are drawn uniformly.
        """
        super().__init__()

        self.splitter = splitter
        self.sources: dict[SplitType, tuple[DatasetSource, ...]] = {
            split: tuple(DatasetSource(**source) for source in sources)
            for split, sources in (
                (SplitType.TRAIN, train_sources),
                (SplitType.VAL, validation_sources),
                (SplitType.TEST, test_sources),
                (SplitType.PREDICT, predict_sources),
            )
            if sources is not None
        }
        self.dataset_factories: dict[SplitType, Callable[..., BaseDataset]] = {
            split: factory
            for split, factory in (
                (SplitType.TRAIN, train_dataset),
                (SplitType.VAL, validation_dataset),
                (SplitType.TEST, test_dataset),
                (SplitType.PREDICT, predict_dataset),
            )
            if factory is not None
        }
        if set(self.sources) != set(self.dataset_factories):
            raise ValueError(
                "Every split needs both its dataset sources and its dataset factory, got "
                f"sources for {sorted(self.sources)} and factories for "
                f"{sorted(self.dataset_factories)}."
            )
        for split, sources in self.sources.items():
            if not len(sources):
                raise ValueError(f"Split {split} needs at least one dataset source.")
            if split != SplitType.TRAIN and any(source.repeat != 1 for source in sources):
                raise ValueError(f"Only the training split can repeat a source, got {split}.")
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
            source.database.taxonomy: source.database.version
            for sources in self.sources.values()
            for source in sources
        }
        if len(taxonomies) > 1:
            raise ValueError(
                "All dataset sources must use the same taxonomy, got different ones in "
                f"{sorted(taxonomies.values())}."
            )

    def databases(self) -> Sequence[BaseDatabase]:
        """
        Every distinct database of every split, in declaration order.

        Returns:
          Sequence[BaseDatabase]: Databases the datamodule reads, one per database hash.
        """
        databases: dict[str, BaseDatabase] = {}
        for sources in self.sources.values():
            for source in sources:
                databases.setdefault(source.database.database_hash, source.database)
        return tuple(databases.values())

    def setup(self, stage: str | None = None) -> None:
        """Build the dataset of every split the stage needs.

        Each database of a split is read and split once, and every source of the split gets a
        dataset of its own carrying that database's records, root directory and supervision.
        The datasets of a split are then concatenated in declaration order.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict') or
                ``None`` to prepare all splits.
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
            if split not in self.sources:
                logger.info(f"No dataset source declared for split {split}, skipping it.")
                continue

            datasets = []
            for source in self.sources[split]:
                database = source.database
                if database.database_hash not in split_records:
                    logger.info(f"Splitting the records of database {database.version}...")
                    split_records[database.database_hash] = self.splitter.split_by_polars_dataframe(
                        dataset_records_dataframe=database.load_polars_scenario_dataframe(),
                        scenarios=database.scenarios,
                    )
                records = split_records[database.database_hash][_RECORD_SPLIT[split]]
                logger.info(
                    f"Serving {len(records)} records of database {database.version} to split "
                    f"{split}, det3d {source.det3d}, seg3d {source.seg3d}, "
                    f"repeated {source.repeat} times."
                )
                datasets.append(
                    self.dataset_factories[split](
                        database_root_path=str(database.root_path),
                        dataset_records_dataframe=records,
                        lidar_intensity_scale=database.lidar_intensity_scale,
                        det3d_supervised=source.det3d,
                        seg3d_supervised=source.seg3d,
                    )
                )

            self.datasets[split] = ConcatDataset(
                datasets=datasets, repeats=[source.repeat for source in self.sources[split]]
            )
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
        """
        if split not in self.datasets:
            raise ValueError(
                f"Split {split} has no dataset. Declare its sources and its dataset factory."
            )
        if config is None:
            raise ValueError(f"Split {split} has no dataloader settings.")

        dataset = self.datasets[split]
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
