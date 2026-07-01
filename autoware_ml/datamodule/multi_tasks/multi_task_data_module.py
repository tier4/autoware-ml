import logging

import lightning as L
from torch.utils.data import DataLoader

from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.datamodule.base import DataLoaderConfig
from autoware_ml.datamodule.multi_tasks.multi_task_base_dataset import (
    MultiTaskBaseDataset,
)
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)


class MultiTaskDataModule(L.LightningDataModule):
    """Base LightningDataModule for multi-task learning that can be shared by multiple datasets."""

    def __init__(
        self,
        database: DatabaseInterface,
        splitter: SplitterInterface,
        train_dataset: MultiTaskBaseDataset | None,
        validation_dataset: MultiTaskBaseDataset | None,
        test_dataset: MultiTaskBaseDataset | None,
        predict_dataset: MultiTaskBaseDataset | None,
        train_dataloader: DataLoaderConfig | None,
        validation_dataloader: DataLoaderConfig | None,
        test_dataloader: DataLoaderConfig | None,
        predict_dataloader: DataLoaderConfig | None,
    ) -> None:
        super().__init__()

        self.database = database
        self.splitter = splitter
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.train_dataloader_config = train_dataloader
        self.validation_dataloader_config = validation_dataloader
        self.test_dataloader_config = test_dataloader
        self.predict_dataloader_config = predict_dataloader

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage.

        This unified implementation handles all stages and eliminates
        code duplication. It maps stages to dataset splits
        and creates datasets using _create_dataset().

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict') or
                ``None`` to prepare all splits.
        """
        # 1) Load the dataset records dataframe from the database
        dataset_records_dataframe = self.database.load_polars_scenario_dataframe()

        # 2) Splitter to split the dataset records into train, validation, and test splits
        logger.info("Splitting dataset records into train, validation, and test splits...")
        split_dataset_dataframes = self.splitter.split_by_polars_dataframe(
            dataset_records_dataframe=dataset_records_dataframe,
            scenarios=self.database.scenarios,
        )
        logger.info("Finished splitting dataset records into train, validation, and test splits.")

        # 3) Assign the split dataset records to the corresponding datasets based on the stage
        logger.info(
            f"Assigning split dataset records to the corresponding datasets"
            f" based on the stage: {stage}..."
        )

        # Define a mapping from stage to the corresponding datasets and their split types
        # stage: [{dataset: split_type}]
        stage_to_datasets = {
            None: [
                (self.train_dataset, SplitType.TRAIN),
                (self.validation_dataset, SplitType.VAL),
                (self.test_dataset, SplitType.TEST),
                (self.predict_dataset, SplitType.PREDICT),
            ],
            "fit": [
                (self.train_dataset, SplitType.TRAIN),
                (self.validation_dataset, SplitType.VAL),
            ],
            "validate": [(self.validation_dataset, SplitType.VAL)],
            "test": [(self.test_dataset, SplitType.TEST)],
            "predict": [(self.predict_dataset, SplitType.PREDICT)],
        }

        stage_datasets = stage_to_datasets.get(stage, [])
        for dataset, split_type in stage_datasets:
            if dataset is not None:
                dataset.assign_dataset_records(split_dataset_dataframes[split_type])
            else:
                logger.warning(
                    f"Dataset for split type {split_type} is not set. Skipping assignment of dataset records."
                )

        logger.info(
            f"Finished assigning split dataset records to the corresponding datasets"
            f" based on the stage: {stage}..."
        )

    def prepare_data(self) -> None:
        """
        Prepare the data for the multi-task learning.
        This method is called only in a single process from a main node.
        """
        logger.info("Preparing data for multi-task learning...")
        # Process the scenario records and create caches for the database if needed.
        self.database.process_scenario_records()
        logger.info("Finished preparing data for multi-task learning.")

    def train_dataloader(self):
        """Create and validate dataloader for training."""
        if self.train_dataset is None:
            raise ValueError(
                "Train dataset is not set. Please set the train dataset before calling train_dataloader()."
            )
        if self.train_dataloader_config is None:
            raise ValueError(
                "Train dataloader config is not set. Please set the train dataloader config before calling train_dataloader()."
            )

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_dataloader_config.batch_size,
            shuffle=self.train_dataloader_config.shuffle,
            num_workers=self.train_dataloader_config.num_workers,
            pin_memory=self.train_dataloader_config.pin_memory,
            drop_last=self.train_dataloader_config.drop_last,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        """Create and validate dataloader for validation."""
        if self.validation_dataset is None:
            raise ValueError(
                "Validation dataset is not set. Please set the validation dataset before calling val_dataloader()."
            )
        if self.validation_dataloader_config is None:
            raise ValueError(
                "Validation dataloader config is not set. Please set the validation dataloader config before calling val_dataloader()."
            )

        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.validation_dataloader_config.batch_size,
            shuffle=self.validation_dataloader_config.shuffle,
            num_workers=self.validation_dataloader_config.num_workers,
            pin_memory=self.validation_dataloader_config.pin_memory,
            drop_last=self.validation_dataloader_config.drop_last,
            collate_fn=self.validation_dataset.collate_fn,
        )

    def test_dataloader(self):
        """Create and validate dataloader for testing."""
        if self.test_dataset is None:
            raise ValueError(
                "Test dataset is not set. Please set the test dataset before calling test_dataloader()."
            )
        if self.test_dataloader_config is None:
            raise ValueError(
                "Test dataloader config is not set. Please set the test dataloader config before calling test_dataloader()."
            )

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_dataloader_config.batch_size,
            shuffle=self.test_dataloader_config.shuffle,
            num_workers=self.test_dataloader_config.num_workers,
            pin_memory=self.test_dataloader_config.pin_memory,
            drop_last=self.test_dataloader_config.drop_last,
            collate_fn=self.test_dataset.collate_fn,
        )

    def predict_dataloader(self):
        """Create and validate dataloader for prediction."""
        if self.predict_dataset is None:
            raise ValueError(
                "Predict dataset is not set. Please set the predict dataset before calling predict_dataloader()."
            )
        if self.predict_dataloader_config is None:
            raise ValueError(
                "Predict dataloader config is not set. Please set the predict dataloader config before calling predict_dataloader()."
            )

        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.predict_dataloader_config.batch_size,
            shuffle=self.predict_dataloader_config.shuffle,
            num_workers=self.predict_dataloader_config.num_workers,
            pin_memory=self.predict_dataloader_config.pin_memory,
            drop_last=self.predict_dataloader_config.drop_last,
            collate_fn=self.predict_dataset.collate_fn,
        )
