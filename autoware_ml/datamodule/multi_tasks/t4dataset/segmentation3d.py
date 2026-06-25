import polars as pl

from autoware_ml.datamodule.multi_tasks.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_samples import MultiTaskGTSample


class T4Segmentation3DTask(BaseDatasetTask):
    """
    Dataset task for 3D segmentation in the T4 dataset.
    This class defines how to process the dataset records for 3D segmentation in the T4 dataset and retrieve the necessary information for training and evaluation.
    """

    def __init__(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Initialize the T4Segmentation3DTask class.
        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed for 3D segmentation in the T4 dataset.
        """
        super().__init__(dataset_records_dataframe=dataset_records_dataframe)

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        return "T4Segmentation3DTask"

    def get_data_sample(self, idx: int) -> MultiTaskGTSample:
        """
        Process the dataset records dataframe for 3D segmentation in the T4 dataset.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed.
          idx: Index of the specific record to be processed.

        Returns:
          MultiTaskDataRow: Processed multi-task data row for 3D segmentation in the T4 dataset.
        """
        # TODO (Kok Seang): Implement the data retrieval for 3D segmentation in the T4 dataset
        # based on the dataset records dataframe and the given index.
        return MultiTaskGTSample(
            lidar_point_cloud_data_row=None,
            point_cloud_features=None,
            detection3d_data_row=None,
            segmentation3d_data_row=None,
        )
