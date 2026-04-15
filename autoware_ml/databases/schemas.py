from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

from autoware_ml.common.enums.enums import Box3DFieldIndex

__BOX_3D_FIELD_LENGTH = len(Box3DFieldIndex)


class DatasetTableColumn(NamedTuple):
    """
    Annotation table column.

    Attributes:
      name: Name of the column.
      dtype: Data type of the column.
    """

    name: str
    dtype: pl.DataType


@dataclass(frozen=True)
class DatasetTableSchema:
    """
    Annotation table schema.

    Attributes:
      SCENARIO_ID: Scenario ID column.
      SAMPLE_ID: Sample ID column.
      SAMPLE_INDEX: Sample index column.
      LOCATION: Location column.
      VEHICLE_TYPE: Vehicle type column.
      SCENARIO_NAME: Scenario name column.

      # LiDAR Schema
      LIDAR_FRAME_ID: Lidar frame ID column.
      LIDAR_SENSOR_ID: Lidar sensor ID column.
      LIDAR_SENSOR_CHANNEL_NAME: Lidar sensor channel name column.
      LIDAR_POINTCLOUD_PATH: Lidar pointcloud path column.
      LIDAR_POINTCLOUD_SOURCE_PATH: Lidar pointcloud source path column.
      LIDAR_POINTCLOUD_NUM_FEATURES: Lidar pointcloud num features column.
      LIDAR_SENSOR_TO_EGO_POSE_MATRIX: Lidar sensor to ego pose matrix column.
      LIDAR_FRAME_EGO_POSE_TO_GLOBAL_MATRIX: Lidar frame ego pose to global matrix column.

      # Multisweep LiDAR Schema
      LIDAR_SWEEP_FRAME_IDS: Lidar sweep frame IDs column.
      LIDAR_SWEEP_TIMESTAMPS_SECONDS: Lidar sweep timestamps in seconds column.
      LIDAR_SWEEP_POINTCLOUDS_PATHS: Lidar sweep pointclouds paths column.
      LIDAR_SWEEP_FRAME_EGO_POSE_TO_GLOBAL_MATRICES: Lidar sweep frame ego pose to global matrices column.
      LIDAR_SENSOR_TO_LIDAR_SWEEP_MATRICES: Lidar sensor to lidar sweep matrices column.

      # Lidar Sources Schema
      LIDAR_SOURCE_CHANNEL_NAMES: Lidar source channel names column.
      LIDAR_SOURCE_SENSOR_TOKENS: Lidar source sensor tokens column.
      LIDAR_SOURCE_TRANSLATIONS: Lidar source translations column.
      LIDAR_SOURCE_ROTATIONS: Lidar source rotations column.

      # Lidarseg Schema
      LIDARSEG_PTS_SEMANTIC_MASK_PATH: Lidarseg pts semantic mask path column.

      # Category Schema
      CATEGORY_NAMES: Category names column.
      CATEGORY_INDICES: Category indices column.

      # 3D Bounding Boxes Schema
      BOXES_3D_FIELDS: Boxes 3D fields column.
      BOXED_3D_DATASET_LABEL_NAMES: Boxed 3d dataset label names column.
      BOXED_3D_LABEL_NAMES: Boxed 3d label names column.
      BOXED_3D_LABEL_INDICES: Boxed 3d label indices column.
      BOXES_3D_INSTANCE_IDS: Boxes 3d instance IDs column.
    """

    # Basic Schema
    SCENARIO_ID = DatasetTableColumn("scenario_id", pl.String)
    SAMPLE_ID = DatasetTableColumn("sample_id", pl.String)
    SAMPLE_INDEX = DatasetTableColumn("sample_index", pl.Int32)
    TIMESTAMP_SECONDS = DatasetTableColumn("timestamp_seconds", pl.Float64)
    LOCATION = DatasetTableColumn("location", pl.String)
    VEHICLE_TYPE = DatasetTableColumn("vehicle_type", pl.String)
    SCENARIO_NAME = DatasetTableColumn("scenario_name", pl.String)

    # LiDAR Schema
    LIDAR_FRAME_ID = DatasetTableColumn("lidar_frame_id", pl.String)
    LIDAR_SENSOR_ID = DatasetTableColumn("lidar_sensor_id", pl.String)
    LIDAR_SENSOR_CHANNEL_NAME = DatasetTableColumn("lidar_sensor_channel_name", pl.String)
    LIDAR_POINTCLOUD_PATH = DatasetTableColumn("lidar_pointcloud_path", pl.String)
    LIDAR_POINTCLOUD_SOURCE_PATH = DatasetTableColumn("lidar_pointcloud_source_path", pl.String)
    LIDAR_POINTCLOUD_NUM_FEATURES = DatasetTableColumn("lidar_pointcloud_num_features", pl.Int32)
    LIDAR_SENSOR_TO_EGO_POSE_MATRIX = DatasetTableColumn(
        "lidar_sensor_to_ego_pose_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )
    LIDAR_FRAME_EGO_POSE_TO_GLOBAL_MATRIX = DatasetTableColumn(
        "lidar_frame_ego_pose_to_global_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )

    # Multisweep Lidar Schema, they are stored in a list
    LIDAR_SWEEP_FRAME_IDS = DatasetTableColumn("lidar_sweep_frame_ids", pl.List(pl.String))
    LIDAR_SWEEP_TIMESTAMPS_SECONDS = DatasetTableColumn(
        "lidar_sweep_timestamps_seconds", pl.List(pl.Float64)
    )
    LIDAR_SWEEP_POINTCLOUDS_PATHS = DatasetTableColumn(
        "lidar_sweep_pointclouds_paths", pl.List(pl.String)
    )
    LIDAR_SWEEP_FRAME_EGO_POSE_TO_GLOBAL_MATRICES = DatasetTableColumn(
        "lidar_sweep_frame_ego_pose_to_global_matrices", pl.List(pl.Array(pl.Float32, shape=(4, 4)))
    )
    LIDAR_SENSOR_TO_LIDAR_SWEEP_MATRICES = DatasetTableColumn(
        "lidar_sensor_to_lidar_sweep_matrices", pl.List(pl.Array(pl.Float32, shape=(4, 4)))
    )

    # Lidar Sources Schema
    LIDAR_SOURCE_CHANNEL_NAMES = DatasetTableColumn(
        "lidar_source_channel_names", pl.List(pl.String)
    )
    LIDAR_SOURCE_SENSOR_TOKENS = DatasetTableColumn(
        "lidar_source_sensor_tokens", pl.List(pl.String)
    )
    LIDAR_SOURCE_TRANSLATIONS = DatasetTableColumn(
        "lidar_source_translations", pl.List(pl.Array(pl.Float32, shape=(3,)))
    )
    LIDAR_SOURCE_ROTATIONS = DatasetTableColumn(
        "lidar_source_rotations", pl.List(pl.Array(pl.Float32, shape=(3, 3)))
    )

    # Lidarseg Schema, and it can be nullable by default
    LIDARSEG_PTS_SEMANTIC_MASK_PATH = DatasetTableColumn(
        "lidarseg_pts_semantic_mask_path", pl.String
    )

    # {class_name: class index}, and it's now two columns
    CATEGORY_NAMES = DatasetTableColumn("category_names", pl.List(pl.String))
    CATEGORY_INDICES = DatasetTableColumn("category_indices", pl.List(pl.Int32))

    # 3D Bounding Boxes Schema, it's a list of Array(Float32, shape=(__BOX_3D_FIELD_LENGTH, ))
    BOXES_3D_ARRAYS = DatasetTableColumn(
        "boxes_3d_arrays", pl.List(pl.Array(pl.Float32, shape=(__BOX_3D_FIELD_LENGTH,)))
    )
    BOXED_3D_DATASET_LABEL_NAMES = DatasetTableColumn(
        "boxed_3d_dataset_label_names", pl.List(pl.String)
    )
    BOXED_3D_LABEL_NAMES = DatasetTableColumn("boxed_3d_label_names", pl.List(pl.String))
    BOXED_3D_LABEL_INDICES = DatasetTableColumn("boxed_3d_label_indices", pl.List(pl.Int32))
    BOXES_3D_INSTANCE_IDS = DatasetTableColumn("boxes_3d_instance_ids", pl.List(pl.String))
    BOXES_3D_VALID = DatasetTableColumn("boxes_3d_valid", pl.List(pl.Boolean))
    BOXES_3D_ATTRIBUTES = DatasetTableColumn("boxes_3d_attributes", pl.List(pl.String))
    BOXES_3D_NUM_LIDAR_POINTCLOUDS = DatasetTableColumn(
        "boxes_3d_num_lidar_pointclouds", pl.List(pl.Int32)
    )
    BOXED_3D_NUM_RADAR_POINTCLOUDS = DatasetTableColumn(
        "boxed_3d_num_radar_pointclouds", pl.List(pl.Int32)
    )

    @classmethod
    def to_polars_schema(cls) -> pl.Schema:
        """
        Convert the dataset table schema to a Polars schema.

        Returns:
          pl.Schema: Polars schema.
        """

        return pl.Schema(
            {
                v.name: v.dtype
                for k, v in cls.__dict__.items()
                if not k.startswith("__") and isinstance(v, DatasetTableColumn)
            }
        )


class DatasetRecord(BaseModel):
    """
    Data class to save a record for each column in the annotation table.

    Attributes:
      # Basic Metadata
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      location: Location of the vehicle.
      vehicle_type: Type of the vehicle.

      # LiDAR Metadata
      lidar_frame_id: Lidar frame ID.
      lidar_sensor_id: Lidar sensor ID.
      lidar_sensor_channel_name: Lidar sensor channel name.
      lidar_pointcloud_path: Lidar pointcloud path.
      lidar_pointcloud_source_path: Lidar pointcloud source path.
      lidar_pointcloud_num_features: Lidar pointcloud num features.
      lidar_sensor_to_ego_pose_matrix: Lidar sensor to ego pose matrix.
      lidar_frame_ego_pose_to_global_matrix: Lidar frame ego pose to global matrix.

      # Multisweep LiDAR Metadata
      lidar_sweep_frame_ids: List of lidar sweep frame IDs.
      lidar_sweep_timestamps_seconds: List of lidar sweep timestamps in seconds.
      lidar_sweep_pointclouds_paths: List of lidar sweep pointclouds paths.
      lidar_sweep_frame_ego_pose_to_global_matrices: List of lidar sweep frame ego pose to global matrices.
      lidar_sensor_to_lidar_sweep_matrices: List of lidar sensor to lidar sweep matrices.

      # Lidar Sources Metadata
      lidar_source_channel_names: List of lidar source channel names.
      lidar_source_sensor_tokens: List of lidar source sensor tokens.
      lidar_source_translations: List of lidar source translations.
      lidar_source_rotations: List of lidar source rotations.

      # Lidarseg Metadata
      lidarseg_pts_semantic_mask_path: Lidarseg pts semantic mask path.

      # Category Metadata
      category_names: List of category names.
      category_indices: List of category indices.

      # 3D Bounding Boxes Metadata
      boxes_3d_fields: List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y.
      boxed_3d_dataset_label_names: List of dataset label names for each 3d box (N)
      boxed_3d_label_names: List of label names for each 3d box after mapping (N)
      boxed_3d_label_indices: List of label indices for each 3d box after mapping (N)
      boxes_3d_instance_ids: List of instance IDs for each 3d box (N)
    """

    # Set model config to frozen
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # Basic Metadata
    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    location: str
    vehicle_type: str
    scenario_name: str

    # LiDAR Metadata
    lidar_frame_id: str
    lidar_sensor_id: str
    lidar_sensor_channel_name: str
    lidar_pointcloud_path: str
    lidar_pointcloud_source_path: str | None
    lidar_pointcloud_num_features: int
    lidar_sensor_to_ego_pose_matrix: npt.NDArray[np.float32]  # (4, 4)
    lidar_frame_ego_pose_to_global_matrix: npt.NDArray[
        np.float32
    ]  # Ego pose (keyframe: lidar calibrated sensor) to global matrix from the selected lidar sensor (4, 4)

    # Multisweep Lidar Metadata
    lidar_sweep_frame_ids: Sequence[str]
    lidar_sweep_timestamps_seconds: Sequence[float]
    lidar_sweep_pointclouds_paths: Sequence[str]
    lidar_sweep_frame_ego_pose_to_global_matrices: Sequence[npt.NDArray[np.float32]]  # (4, 4)
    lidar_sensor_to_lidar_sweep_matrices: Sequence[npt.NDArray[np.float32]]  # (4, 4)

    # Lidar Sources Metadata
    lidar_source_channel_names: Sequence[str]
    lidar_source_sensor_tokens: Sequence[str]
    lidar_source_translations: Sequence[npt.NDArray[np.float32]]
    lidar_source_rotations: Sequence[npt.NDArray[np.float32]]

    # Lidarseg Metadata
    lidarseg_pts_semantic_mask_path: str | None

    # Category names and indices
    category_names: Sequence[str]
    category_indices: Sequence[int]

    # 3D bounding boxes schema
    boxes_3d_arrays: Sequence[npt.NDArray[np.float32]]  # (N, length of Box3DFieldIndex)
    boxed_3d_dataset_label_names: Sequence[str]  # Label names for each 3d box before mapping (N)
    boxed_3d_label_names: Sequence[str]  # Label names for each 3d box after mapping (N)
    boxed_3d_label_indices: Sequence[int]  # Label indices for each 3d box after mapping (N)
    boxes_3d_instance_ids: Sequence[str]  # Instance IDs for each 3d box (N)
    boxes_3d_valid: Sequence[bool]  # Valid flag for each 3d box (N)
    boxes_3d_attributes: Sequence[str]  # Attributes for each 3d box (N)
    boxes_3d_num_lidar_pointclouds: Sequence[int]  # Number of lidar pointclouds for each 3d box (N)
    boxed_3d_num_radar_pointclouds: Sequence[int]  # Number of radar pointclouds for each 3d box (N)
