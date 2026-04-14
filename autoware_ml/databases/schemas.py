from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict


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
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      location: Location of the vehicle.
      vehicle_type: Type of the vehicle.
      bbox_3d: List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y.
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
    # lidarseg Metadata
