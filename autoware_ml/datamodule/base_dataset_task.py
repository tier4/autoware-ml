from abc import ABC, abstractmethod
import logging
from pathlib import Path

import polars as pl

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


logger = logging.getLogger(__name__)


class BaseDatasetTask(ABC):
    """
    Abstract interface for dataset tasks that defines how a task-specific dataset should be
    implemented when retrieving data.
    """

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

    def __str__(self) -> str:
        """
        Name of the task, its class name unless a task overrides it.

        Returns:
          str: Name of the task.
        """
        return type(self).__name__

    def log_dataset_info(self) -> None:
        """
        Log one line on the records the task reads.

        The dataset calls this on rank zero every time it is bound to records, so it must stay
        cheap: the default reports the record count, and a task overriding it should only touch
        the columns it summarises.
        """
        logger.info(f"{self}: {len(self.dataset_records_dataframe)} records.")
