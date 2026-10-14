from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class BaseDatasetTask(ABC):
    """
    Abstract interface for dataset tasks that defines how a task-specific dataset should be
    implemented when retrieving data.
    """

    def __init__(self, database_root_path: str, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Initialize the dataset task.
        """
        self.database_root_path = Path(database_root_path)
        self.dataset_records_dataframe = self.select_columns(dataset_records_dataframe)

    @abstractmethod
    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the columns of the records the task reads.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with the columns of the task.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        raise NotImplementedError("Dataset type must define __str__!")

    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Process the dataset records dataframe for the specific task.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed.
          idx: Index of the specific record to be processed.

        Returns:
          ModelGTSample: Multi-task data for training/inference with ground truths
            from a sample.
        """
        raise NotImplementedError("Dataset type must define get_data_sample()!")

    def log_dataset_info(self) -> None:
        """
        Log and print dataset information for the specific task.
        """
        raise NotImplementedError("Dataset type must define log_dataset_info()!")
