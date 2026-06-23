from typing import Sequence

import Lightning as L
from torch.utils.data import DataLoader

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.types.tasks import TaskType
from autoware_ml.datamodule.base import DataLoaderConfig
from autoware_ml.datamodule.multi_tasks.multi_task_dataset_interface import (
    MultiTaskDatasetInterface,
)
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface


class MultiTaskDataModule(L.LightningDataModule):
    """
    Base LightningDataModule for multi-task learning that can be shared by multiple datasets.
    """

    def __init__(
        self,
        tasks: Sequence[TaskType],
        database: BaseDatabase,
        splitter: SplitterInterface,
        train_dataset: MultiTaskDatasetInterface | None,
        validation_dataset: MultiTaskDatasetInterface | None,
        test_dataset: MultiTaskDatasetInterface | None,
        train_dataloader: DataLoaderConfig | None,
        validation_dataloader: DataLoaderConfig | None,
        test_dataloader: DataLoaderConfig | None,
    ) -> None:
        super().__init__()

        self.tasks = tasks
        self.database = database
        self.splitter = splitter
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.train_dataloader_config = train_dataloader
        self.validation_dataloader_config = validation_dataloader
        self.test_dataloader_config = test_dataloader

        self.tasks = tasks

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage.

        This unified implementation handles all stages and eliminates
        code duplication. It maps stages to dataset splits
        and creates datasets using _create_dataset().

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict') or
                ``None`` to prepare all splits.
        """
        # Define stage to splits mapping
        stage_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": ["predict"],
        }

        # Get splits for this stage
        splits = ["train", "val", "test", "predict"] if stage is None else stage_splits[stage]

        # Create datasets for required splits
        for split in splits:
            if getattr(self, f"{split}_dataset") is not None:
                continue
            transforms = getattr(self, f"{split}_transforms")
            dataset = self._create_dataset(split, transforms)
            setattr(self, f"{split}_dataset", dataset)

    def prepare_data(self) -> None:
        """
        Prepare the data for the multi-task learning.
        This method is called only from a single process from a main node.
        """
        # Process the scenario records and create caches for the database if needed.
        self.database.process_scenario_records()

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
        )
