# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset task reading the 3D boxes of a record."""

import logging
from collections.abc import Sequence

import numpy as np
from jaxtyping import Int64
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.types.geometry import Box3DFieldIndex, Box3DCenterCoordinateType

logger = logging.getLogger(__name__)


def flatten_box_fields(
    records: pl.DataFrame, fields: Sequence[str]
) -> tuple[Int64[np.ndarray, " num_records_plus_1"], pl.DataFrame]:
    """
    Gather some fields of every box of every record, without reading the others.

    Args:
      records: Records with their boxes.
      fields: Names of the box fields to gather.

    Returns:
      tuple: Offset of every record's boxes (one more than the records) and the fields of every
        box, in record order.

    Raises:
      ValueError: If the gathered boxes disagree with the box counts of the records.
    """
    boxes = DatasetTableSchema.BOXES_3D.name
    counts = records[boxes].list.len().fill_null(0).to_numpy().astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    flat_fields = (
        records.select(
            pl.col(boxes).list.eval(pl.element().struct.field(field)).alias(field)
            for field in fields
        )
        .explode(list(fields))
        .drop_nulls()
    )
    if flat_fields.height != offsets[-1]:
        raise ValueError(f"Gathered {flat_fields.height} boxes but the records list {offsets[-1]}.")
    return offsets, flat_fields


def box_instance_ids(tokens: pl.Series) -> Int64[np.ndarray, " num_boxes"]:
    """
    Instance id of every box, the hashed instance token, one per box.

    Args:
      tokens: Instance token of every box.

    Returns:
      The hashed instance token of every box.
    """
    # A track repeats its token in every frame, so every distinct token is hashed once
    unique_tokens = tokens.unique()
    return (
        tokens.replace_strict(
            unique_tokens,
            pl.Series(hash_instance_ids(unique_tokens.to_list()), dtype=pl.Int64),
            return_dtype=pl.Int64,
        )
        .to_numpy()
        .astype(np.int64, copy=False)
    )


class Detection3DTask(BaseDatasetTask):
    """
    Read the 3D boxes of a record and its cone and barrier annotation flag.

    The records store the fine label name of every box, the taxonomy resolves it to a class
    index when a sample is loaded.
    """

    COLUMNS = (
        DatasetTableSchema.BOXES_3D.name,
        DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name,
    )

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame,
        taxonomy: LabelTaxonomy,
    ) -> None:
        """
        Initialize the detection task.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus.
          taxonomy: Taxonomy that maps the fine label name of a box to its class index.

        Raises:
          KeyError: If a box of the records carries a fine name the taxonomy does not map.
        """
        super().__init__(
            database_root_path=database_root_path,
            dataset_records_dataframe=dataset_records_dataframe,
        )
        self.taxonomy = taxonomy
        instance_id = Box3DDatasetSchema.BOX3D_INSTANCE_ID.name
        label = Box3DDatasetSchema.BOX3D_LABEL_NAME.name
        self.box_offsets, box_fields = flatten_box_fields(
            dataset_records_dataframe, [instance_id, label]
        )
        self.box_instance_ids = box_instance_ids(box_fields[instance_id])
        self.class_indices = {
            fine_name: taxonomy.class_index(fine_name)
            for fine_name in box_fields[label].unique().to_list()
        }

    def hash_repr(self) -> str:
        """
        Canonical text of the task settings.

        Returns:
          str: The taxonomy the task maps its labels with.
        """
        return f"{type(self).__name__}(taxonomy={self.taxonomy})"

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the box column and the cone and barrier flag column of the records.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with their boxes and their flag.

        Raises:
          ValueError: If a column is missing. Every record table carries both, so a table
            without the flag was not built by this version of the database.
        """
        missing = [
            column for column in self.COLUMNS if column not in dataset_records_dataframe.columns
        ]
        if missing:
            raise ValueError(
                f"{self} needs the columns {list(self.COLUMNS)} of the record table, missing "
                f"{missing}. Rebuild the database cache with this version."
            )
        return dataset_records_dataframe.select(list(self.COLUMNS))

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        return "Detection3DTask"

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
        gt_bboxes_label_names = selected_row.field(
            Box3DDatasetSchema.BOX3D_LABEL_NAME.name
        ).to_list()
        gt_bboxes_labels = np.asarray(
            [self.class_indices[name] for name in gt_bboxes_label_names], dtype=np.int32
        )
        gt_bboxes_num_lidar_points = (
            selected_row.field(Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTS.name)
            .to_numpy()
            .astype(np.int32, copy=False)
        )
        gt_bboxes_attributes = selected_row.field(
            Box3DDatasetSchema.BOX3D_ATTRIBUTES.name
        ).to_list()
        gt_bboxes_instance_ids = self.box_instance_ids[
            self.box_offsets[idx] : self.box_offsets[idx + 1]
        ].copy()

        # The boxes are served as the records store them, non finite values included: a box
        # with no annotated velocity carries a non finite one. BBoxesPhysicalFilter in the
        # transform pipeline zeroes such velocities and drops the boxes whose geometry cannot
        # become a target. Boxes with no lidar points are kept, the training filters and the
        # metrics apply the point count.
        if not len(gt_bboxes_3d):
            gt_bboxes_3d = np.zeros((0, len(Box3DFieldIndex)), dtype=np.float32)
            gt_bboxes_labels = np.zeros((0,), dtype=np.int32)
            gt_bboxes_num_lidar_points = np.zeros((0,), dtype=np.int32)
            gt_bboxes_instance_ids = np.zeros((0,), dtype=np.int64)

        detection3d_bboxes_3d = LidarBBoxes3D.from_numpy(
            bbox_params=gt_bboxes_3d,
            bbox_labels=gt_bboxes_labels,
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
            bbox_label_names=gt_bboxes_label_names,
            bbox_num_lidar_points=gt_bboxes_num_lidar_points,
            bbox_instance_ids=gt_bboxes_instance_ids,
            bbox_attributes=gt_bboxes_attributes,
        )

        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=detection3d_bboxes_3d,
            segmentation3d_gt_sample=None,
            detection3d_traffic_cone_barrier_bbox_status=(
                self._get_traffic_cone_barrier_bbox_status(idx)
            ),
        )

    def _get_traffic_cone_barrier_bbox_status(self, idx: int) -> bool | None:
        """
        Read whether traffic cones and barriers are annotated in the given record.

        Args:
          idx: Index of the record.

        Returns:
          bool | None: The flag, or None when the scenario of the record has no status.
        """
        status = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name
        )
        return None if status is None else bool(status)

    def log_dataset_info(self) -> None:
        """
        Log the number of boxes per class, before and after dropping the boxes without lidar
        points.

        Only the label and the lidar point count are pulled out of the box structs, so the
        count never materialises the full box table.
        """
        boxes = DatasetTableSchema.BOXES_3D.name
        label = Box3DDatasetSchema.BOX3D_LABEL_NAME.name
        has_points = "has_lidar_points"
        class_counts = (
            self.dataset_records_dataframe.select(
                pl.col(boxes).list.eval(pl.element().struct.field(label)).alias(label),
                pl.col(boxes)
                .list.eval(
                    pl.element().struct.field(Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTS.name) > 0
                )
                .alias(has_points),
            )
            # A record without boxes contributes no row; a null box list explodes to one null
            # row, which is not a box either.
            .explode(label, has_points, empty_as_null=False)
            .drop_nulls(label)
            .group_by(label)
            .agg(pl.len().alias("count"), pl.col(has_points).sum().alias("valid_count"))
            .sort("count", descending=True)
        )
        class_names = class_counts[label].to_list()
        total_counts = dict(zip(class_names, class_counts["count"].to_list(), strict=True))
        valid_counts = dict(zip(class_names, class_counts["valid_count"].to_list(), strict=True))

        logger.info(f"{self}: {len(self.dataset_records_dataframe)} records.")
        logger.info(f"Number of bboxes per class in the dataset: {total_counts}")
        logger.info(
            f"Number of bboxes after filtering num_lidar_points > 0 per class: {valid_counts}"
        )
