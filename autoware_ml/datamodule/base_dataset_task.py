from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class BaseDatasetTask(ABC):
    """Read the annotations of one task from the record table of a corpus."""

    def __init__(self, database_root_path: str, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Initialize the dataset task.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus.
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

    @abstractmethod
    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Read the annotations of one record.

        Args:
          idx: Index of the record.

        Returns:
          ModelGTSample: Sample holding the annotations of the task.
        """
