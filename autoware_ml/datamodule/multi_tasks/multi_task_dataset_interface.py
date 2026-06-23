from abc import abstractmethod
from typing import Any, Protocol


import polars as pl
from torch.utils.data import Dataset

from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_data_row import MultiTaskDataRow
from autoware_ml.transforms.base import TransformsCompose


class MultiTaskDatasetInterface(Dataset, Protocol):
    """ "
    Multi-task dataset interface that can be shared by multiple databases.
    """

    def __init__(
        self, dataset_records_dataframe: pl.DataFrame | None, transforms: TransformsCompose | None
    ) -> None:
        """
        Initialize the multi-task dataset interface.
        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in the
              multi-task dataset. Accept None if the dataset records
              are not available at initialization.
          transforms: Global transforms to be applied to the dataset records.
        """
        super().__init__()
        self.transforms = transforms
        self.dataset_records_dataframe = dataset_records_dataframe

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed dataset sample.
        """
        multi_task_data_row = self.get_data_row(index)
        return self.apply_transforms(multi_task_data_row)

    @abstractmethod
    def get_data_row(self, index: int) -> MultiTaskDataRow:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            Metadata dictionary consumed by the transform pipeline.
        """
        raise NotImplementedError("Dataset must implement get_data_info")

    def apply_transforms(
        self,
        multi_task_data_row: MultiTaskDataRow,
    ) -> MultiTaskDataRow:
        """Apply a specific transform pipeline to a metadata sample.

        Args:
            multi_task_data_row: MultiTaskDataRow instance.

        Returns:
            Transformed MultiTaskDataRow instance.
        """
        if self.transforms is None:
            return multi_task_data_row
        return self.transforms(multi_task_data_row)
