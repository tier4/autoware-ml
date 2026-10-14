import numpy as np
import logging

import polars as pl
import torch

from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample


# Data type of the raw category index stored for every point of a semantic mask
SEMANTIC_MASK_DTYPE = np.uint8


logger = logging.getLogger(__name__)


class T4Segmentation3DTask(BaseDatasetTask):
    """Read the semantic labels of the current lidar frame of a record."""

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame,
        taxonomy: LabelTaxonomy,
    ) -> None:
        """
        Initialize the segmentation task.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus.
          taxonomy: Taxonomy that maps a category name to its class index.
        """
        super().__init__(
            database_root_path=database_root_path,
            dataset_records_dataframe=dataset_records_dataframe,
        )
        self.taxonomy = taxonomy

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        return self.__class__.__name__

    def log_dataset_info(self) -> None:
        """
        Log the number of records the semantic labels are read from.
        """
        if self.dataset_records_dataframe is None:
            logger.warning("Dataset records dataframe is not available.")
            return
        logger.info(f"{self}: {len(self.dataset_records_dataframe)} records with semantic labels.")

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the lidar frames and the category mapping of the records.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with the columns the semantic labels are read from.
        """
        return dataset_records_dataframe.select(
            [DatasetTableSchema.LIDAR_FRAMES.name, DatasetTableSchema.CATEGORY_MAPPING.name]
        )

    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Read the semantic labels of the current lidar frame.

        The mask stores a raw category index per point. The record maps each index to a name,
        and the taxonomy maps each name to a class index.

        Args:
          idx: Index of the record.

        Returns:
          ModelGTSample: Sample holding the semantic labels of the current frame.
        """
        lidar_frame = LidarFrameDataModel.load_from_dictionary(
            self.dataset_records_dataframe.item(idx, DatasetTableSchema.LIDAR_FRAMES.name)[0]
        )
        if lidar_frame.lidar_pointcloud_semantic_mask_path is None:
            raise ValueError(
                f"The lidar frame {lidar_frame.lidar_frame_id} of record {idx} carries no "
                "semantic mask, so 3D segmentation cannot be supervised on it."
            )

        raw_labels = np.fromfile(
            self.semantic_mask_path(lidar_frame), dtype=SEMANTIC_MASK_DTYPE
        ).astype(np.int64)
        category_mapping = CategoryMappingDataModel.load_from_dictionary(
            self.dataset_records_dataframe.item(idx, DatasetTableSchema.CATEGORY_MAPPING.name)
        )

        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=Segmentation3DGTSample(
                gt_semantic_mask=torch.from_numpy(
                    self.resolve_class_indices(raw_labels, category_mapping)
                ),
                ignore_index=self.taxonomy.ignore_index,
            ),
        )

    def semantic_mask_path(self, lidar_frame: LidarFrameDataModel) -> str:
        """
        Resolve the stored semantic mask path of a lidar frame against the database root.

        Args:
          lidar_frame: Lidar frame carrying a semantic mask.

        Returns:
          str: Path of the mask on this machine.
        """
        return str(
            self.database_root_path / lidar_frame.lidarseg_pointcloud_semantic_mask_relative_path
        )

    def resolve_class_indices(
        self,
        raw_labels: np.ndarray,
        category_mapping: CategoryMappingDataModel,
    ) -> np.ndarray:
        """
        Resolve the raw category index of every point to the class index of the taxonomy.

        Args:
          raw_labels: Raw category index of every point of the current frame.
          category_mapping: Category mapping of the record, naming every raw category index.

        Returns:
          np.ndarray: Class index of every point of the current frame.
        """
        lookup = np.full(
            max(category_mapping.category_indices, default=-1) + 1,
            self.taxonomy.ignore_index,
            dtype=np.int64,
        )
        named = np.zeros(lookup.shape[0], dtype=np.bool_)
        for category_name, category_index in zip(
            category_mapping.category_names, category_mapping.category_indices
        ):
            lookup[category_index] = self.taxonomy.resolve_index(category_name)
            named[category_index] = True

        in_range = (raw_labels >= 0) & (raw_labels < lookup.shape[0])
        known = in_range.copy()
        known[in_range] = named[raw_labels[in_range]]
        if not known.all():
            raise ValueError(
                "The semantic mask carries the raw category indices "
                f"{sorted(set(raw_labels[~known].tolist()))}, which the category mapping of the "
                "record does not name."
            )
        return lookup[raw_labels]
