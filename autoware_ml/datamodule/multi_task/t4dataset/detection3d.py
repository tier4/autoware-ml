import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.datamodule.multi_task.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LiDARBBoxes3D
from autoware_ml.types.geometry import Box3DFieldIndex, Box3DCenterCoordinateType


class T4Detection3DTask(BaseDatasetTask):
    """
    Dataset task for 3D detection in the T4 dataset.
    This class defines how to process the dataset records for 3D detection in the T4 dataset and retrieve the necessary information for training and evaluation.
    """

    def __init__(
        self, dataset_records_dataframe: pl.DataFrame | None, filter_valid_masks: bool = True
    ) -> None:
        """
        Initialize the T4Detection3DTask class.
        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed for 3D detection in the T4 dataset.
          filter_valid_masks: Whether to filter out invalid bounding boxes based on valid_mask.
        """
        super().__init__(dataset_records_dataframe=dataset_records_dataframe)
        self.filter_valid_masks = filter_valid_masks

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
                DatasetTableSchema.BOXES_3D.name,
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
        if self.dataset_records_dataframe is None:
            raise ValueError("Dataset records dataframe is not available.")

        # Retrieve the specific row from the dataset records dataframe based on the given index
        # and the bbox3d column.
        selected_row = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.BOXES_3D.name
        ).struct
        gt_bboxes_3d = (
            selected_row.field(Box3DDatasetSchema.BOX3D_PARAMS.name)
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        gt_bboxes_labels = (
            selected_row.field(Box3DDatasetSchema.BOX3D_LABEL_INDEX.name)
            .to_numpy()
            .astype(np.int32, copy=False)
        )
        gt_bboxes_valid = (
            selected_row.field(Box3DDatasetSchema.BOX3D_VALID.name)
            .to_numpy()
            .astype(np.bool_, copy=False)
        )

        if not len(gt_bboxes_3d):
            gt_bboxes_3d = np.zeros(
                (0, len(Box3DFieldIndex)), dtype=np.float32
            )  # Zero shape of (0, 10) for empty bboxes
            gt_bboxes_labels = np.zeros((0,), dtype=np.int32)
        elif self.filter_valid_masks:
            # Filter out invalid bounding boxes based on the valid mask if filter_valid_masks is True
            gt_bboxes_3d = gt_bboxes_3d[gt_bboxes_valid]
            gt_bboxes_labels = gt_bboxes_labels[gt_bboxes_valid]

        detection3d_bboxes_3d = LiDARBBoxes3D.from_numpy(
            bbox_params=gt_bboxes_3d,
            bbox_labels=gt_bboxes_labels,
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        )

        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            point_cloud_features=None,
            detection3d_gt_bboxes_3d=detection3d_bboxes_3d,
            segmentation3d_gt_sample=None,
        )
