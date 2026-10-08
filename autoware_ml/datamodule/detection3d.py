import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.types.geometry import Box3DFieldIndex, Box3DCenterCoordinateType


class Detection3DTask(BaseDatasetTask):
    """Read the 3D boxes of a record."""

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the box column of the records.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with their boxes.
        """
        return dataset_records_dataframe.select([DatasetTableSchema.BOXES_3D.name])

    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Read the boxes of one record.

        Args:
          idx: Index of the record.

        Returns:
          ModelGTSample: Sample holding the 3D boxes of the record.
        """
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
        gt_bboxes_label_names = selected_row.field(
            Box3DDatasetSchema.BOX3D_LABEL_NAME.name
        ).to_list()
        gt_bboxes_num_lidar_points = (
            selected_row.field(Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTS.name)
            .to_numpy()
            .astype(np.int32, copy=False)
        )
        gt_bboxes_attributes = selected_row.field(
            Box3DDatasetSchema.BOX3D_ATTRIBUTES.name
        ).to_list()

        if not len(gt_bboxes_3d):
            gt_bboxes_3d = np.zeros((0, len(Box3DFieldIndex)), dtype=np.float32)
            gt_bboxes_labels = np.zeros((0,), dtype=np.int32)
            gt_bboxes_num_lidar_points = np.zeros((0,), dtype=np.int32)
        else:
            # A box with no annotated velocity is stored with a non finite one and is served
            # at zero velocity. The size targets are log encoded, so boxes with a non finite
            # geometry or a zero extent are dropped. Boxes with no lidar points are kept, the
            # training filters and the metrics apply the point count.
            finite = np.isfinite(gt_bboxes_3d)
            extents = gt_bboxes_3d[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.YAW]
            valid = finite[:, : Box3DFieldIndex.VELOCITY_X].all(axis=1) & (extents > 0.0).all(
                axis=1
            )
            gt_bboxes_3d = np.where(finite, gt_bboxes_3d, np.float32(0.0))[valid]
            gt_bboxes_labels = gt_bboxes_labels[valid]
            gt_bboxes_label_names = [
                name for name, keep in zip(gt_bboxes_label_names, valid, strict=True) if keep
            ]
            gt_bboxes_num_lidar_points = gt_bboxes_num_lidar_points[valid]
            gt_bboxes_attributes = [
                attributes
                for attributes, keep in zip(gt_bboxes_attributes, valid, strict=True)
                if keep
            ]

        detection3d_bboxes_3d = LidarBBoxes3D.from_numpy(
            bbox_params=gt_bboxes_3d,
            bbox_labels=gt_bboxes_labels,
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
            bbox_label_names=gt_bboxes_label_names,
            bbox_num_lidar_points=gt_bboxes_num_lidar_points,
            bbox_attributes=gt_bboxes_attributes,
        )

        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=detection3d_bboxes_3d,
            segmentation3d_gt_sample=None,
        )
