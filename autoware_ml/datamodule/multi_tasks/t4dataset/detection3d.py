import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.datamodule.multi_tasks.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.multi_tasks.dataclasses.detection3d import Detection3DGTSample
from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_samples import MultiTaskGTSample


class T4Detection3DTask(BaseDatasetTask):
    """
    Dataset task for 3D detection in the T4 dataset.
    This class defines how to process the dataset records for 3D detection in the T4 dataset and retrieve the necessary information for training and evaluation.
    """

    def __init__(self, dataset_records_dataframe: pl.DataFrame | None) -> None:
        """
        Initialize the T4Detection3DTask class.
        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed for 3D detection in the T4 dataset.
        """
        super().__init__(dataset_records_dataframe=dataset_records_dataframe)

    def pre_filter_dataset_records(
        self, dataset_records_dataframe: pl.DataFrame | None
    ) -> pl.DataFrame | None:
        """
        Pre-filter the dataset records dataframe for 3D detection in the T4 dataset.
        This method filters the dataset records dataframe to only include columns related to 3D bounding boxes.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be filtered for 3D detection in the T4 dataset.
        Returns:
          Polars DataFrame of filtered dataset records for 3D detection in the T4 dataset
        """
        if dataset_records_dataframe is None:
            return None

        # Filter the dataset records dataframe to only include columns related to 3D bounding boxes
        filtered_dataset_records_dataframe = dataset_records_dataframe.select(
            [
                DatasetTableSchema.Box3D.name,
                # Add other necessary columns for 3D detection as needed
            ]
        )
        return filtered_dataset_records_dataframe

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        return "T4Detection3DTask"

    def get_data_sample(self, idx: int) -> MultiTaskGTSample:
        """
        Process the dataset records dataframe for 3D detection in the T4 dataset.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed.
          idx: Index of the specific record to be processed.

        Returns:
          MultiTaskDataRow: Processed multi-task data row for 3D detection in the T4 dataset.
        """
        # Retrieve the specific row from the dataset records dataframe based on the given index
        # and the bbox3d column.
        selected_row = self.dataset_records_dataframe.row(idx, named=True)
        gt_bboxes_3d = selected_row[DatasetTableSchema.Box3D.name]
        gt_bboxes_labels = selected_row[Box3DDatasetSchema.Labels.name]

        detection3d_data_row = Detection3DGTSample(
            gt_bboxes_3d=np.asarray(gt_bboxes_3d, dtype=np.float32),
            gt_labels_3d=np.asarray(gt_bboxes_labels, dtype=np.int32),
            # Add other necessary fields for 3D detection as needed
        )

        return MultiTaskGTSample(
            lidar_point_cloud_data_row=None,
            point_cloud_features=None,
            detection3d_data_row=detection3d_data_row,
            segmentation3d_data_row=None,
        )
