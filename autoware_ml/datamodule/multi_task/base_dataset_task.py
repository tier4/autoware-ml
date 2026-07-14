from abc import ABC
from pathlib import Path

import polars as pl

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample


class BaseDatasetTask(ABC):
    """
    Abstract interface for dataset tasks that defines how a task-specific dataset should be
    implemented when retrieving data.
    """

    def __init__(
        self, database_root_path: str, dataset_records_dataframe: pl.DataFrame | None
    ) -> None:
        """
        Initialize the dataset task.
        """
        self.database_root_path = Path(database_root_path)
        self.dataset_records_dataframe = self.pre_filter_dataset_records(dataset_records_dataframe)

    def pre_filter_dataset_records(
        self, dataset_records_dataframe: pl.DataFrame | None
    ) -> pl.DataFrame | None:
        """
        Pre-filter the dataset records dataframe for the specific task.
          For example, if the task is 3D detection, the dataset records dataframe can be
          filtered to only include columns related to 3D bounding boxes.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be filtered.
        """
        if dataset_records_dataframe is None:
            return None
        return dataset_records_dataframe

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        raise NotImplementedError("Dataset type must define __str__!")

    def get_data_sample(self, idx: int) -> MultiTaskGTSample:
        """
        Process the dataset records dataframe for the specific task.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed.
          idx: Index of the specific record to be processed.

        Returns:
          MultiTaskGTSample: Multi-task data for training/inference with ground truths
            from a sample.
        """
        raise NotImplementedError("Dataset type must define get_data_sample()!")
