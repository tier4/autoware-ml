import logging
from types import MappingProxyType

import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.datamodule.multi_tasks.multi_task_base_dataset import (
    MultiTaskBaseDataset,
)
from autoware_ml.datamodule.multi_tasks.base_dataset_task import BaseDatasetTask
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.types.tasks import TaskType


logger = logging.getLogger(__name__)


class MultiTaskT4Dataset(MultiTaskBaseDataset):
    """
    A dataset class that supports multiple tasks.
    It extends MultiTaskDatasetInterface to include implementation of data retrieval for multiple
    tasks in a single interface.
    """

    def __init__(
        self,
        dataset_records_dataframe: pl.DataFrame | None,
        transforms: TransformsCompose | None,
        dataset_tasks: MappingProxyType[TaskType | str, BaseDatasetTask],
    ) -> None:
        """
        Initialize the MultiTaskT4Dataset class.
        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in
            the multi-task dataset.
          transforms: Global transforms to be applied to the dataset records.
          dataset_tasks: Every task dataset that is part of the multi-task dataset, mapped by
            task type.
        """
        super().__init__(dataset_records_dataframe=dataset_records_dataframe, transforms=transforms)
        # Convert the dataset_tasks to TaskType: BaseDatasetTask mapping if the keys are strings
        self.dataset_tasks: MappingProxyType[TaskType, BaseDatasetTask] = MappingProxyType(
            {
                TaskType(key) if isinstance(key, str) else key: value
                for key, value in dataset_tasks.items()
            }
        )
        logger.info(
            f"Initialized MultiTaskT4Dataset with {len(self.dataset_tasks)} "
            f"task datasets: {list(self.dataset_tasks.keys())}."
        )

    def get_data_sample(self, idx: int) -> MultiTaskGTSample:
        """
        Process the dataset records dataframe for multiple tasks in the T4 dataset.

        Args:
          idx: Index of the specific record to be processed.

        Returns:
          MultiTaskGTSample: Processed multi-task data row, mapped by task type.
        """
        data_rows = {}
        for task_type, dataset_task in self.dataset_tasks.items():
            data_rows[task_type] = dataset_task.get_data_sample(idx)

        # Retrieve general data row for the given index from the dataset records dataframe

        # Merge the data rows from different tasks into a single multi-task data row
        return MultiTaskGTSample(**data_rows)

    def get_lidar_pointcloud_data_row(self, idx: int) -> MultiTaskGTSample:
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

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Recursively assign the dataset records dataframe to each task dataset as well and
        perform their pre_filtering .

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        self.dataset_records_dataframe = dataset_records_dataframe
        for dataset_task in self.dataset_tasks.values():
            filtered_dataset_records_dataframe = dataset_task.pre_filter_dataset_records(
                dataset_records_dataframe
            )
            dataset_task.dataset_records_dataframe = filtered_dataset_records_dataframe
