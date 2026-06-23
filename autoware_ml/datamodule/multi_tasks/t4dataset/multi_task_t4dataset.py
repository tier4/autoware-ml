from types import MappingProxyType

import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_data_row import (
    MultiTaskDataRow,
)
from autoware_ml.datamodule.multi_tasks.multi_task_dataset_interface import (
    MultiTaskDatasetInterface,
)
from autoware_ml.datamodule.multi_tasks.base_dataset_task import BaseDatasetTask
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.types.tasks import TaskType


class MultiTaskT4Dataset(MultiTaskDatasetInterface):
    """
    A dataset class that supports multiple tasks.
    It extends MultiTaskDatasetInterface to include implementation of data retrieval for multiple
    tasks in a single interface.
    """

    def __init__(
        self,
        dataset_records_dataframe: pl.DataFrame,
        transforms: TransformsCompose | None,
        task_datasets: MappingProxyType[TaskType, BaseDatasetTask],
    ) -> None:
        """
        Initialize the MultiTaskT4Dataset class.
        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in
            the multi-task dataset.
          transforms: Global transforms to be applied to the dataset records.
          task_datasets: Every task dataset that is part of the multi-task dataset, mapped by
            task type.
        """
        super().__init__(dataset_records_dataframe=dataset_records_dataframe, transforms=transforms)
        self.task_datasets = task_datasets

    def get_data_row(self, idx: int) -> MultiTaskDataRow:
        """
        Process the dataset records dataframe for multiple tasks in the T4 dataset.

        Args:
          idx: Index of the specific record to be processed.

        Returns:
          MultiTaskDataRow: Processed multi-task data row, mapped by task type.
        """
        data_rows = {}
        for task_type, task_dataset in self.task_datasets.items():
            data_rows[task_type] = task_dataset.get_data_row(idx)

        # Retrieve general data row for the given index from the dataset records dataframe

        # Merge the data rows from different tasks into a single multi-task data row
        return MultiTaskDataRow(**data_rows)

    def get_lidar_pointcloud_data_row(self, idx: int) -> MultiTaskDataRow:
        """
        Retrieve the lidar point cloud data row for the given index.

        Args:
          idx: Index of the specific record to be processed.
        """
        # Retrieve the lidar point cloud data row for the given index from the dataset records dataframe
        lidar_pointcloud_data = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.LIDAR_FRAMES.name
        )

        # lidar_pointcloud_data_row = LiDARPointCloudDataRow(lidar_pointcloud_data)
        return lidar_pointcloud_data
