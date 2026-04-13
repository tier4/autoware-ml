import logging

from typing import Sequence, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from t4_devkit.dataclass.box import Box3D

from autoware_ml.databases.schemas import DatasetRecord

logger = logging.getLogger(__name__)


class T4SampleRecordBasicInfo(BaseModel):
    """
    Basic information of a T4 sample record.

    Attributes:
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      timestamp_seconds: Timestamp in seconds.
      scenario_name: Scenario name.
      location: Location.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    scenario_name: str
    location: str | None = None
    vehicle_type: str | None = None


class T4SampleRecordLidarInfo(BaseModel):
    """
    Lidar information of a T4 sample record.

    Attributes:
      lidar_frame_id: Lidar frame ID.
      lidar_sensor_id: Lidar sensor ID.
      lidar_sensor_channel_name: Lidar sensor channel name.
      lidar_pointcloud_path: Lidar pointcloud path.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    lidar_frame_id: str
    lidar_sensor_id: str
    lidar_sensor_channel_name: str
    lidar_pointcloud_path: str
    lidar_pointcloud_num_features: int
    lidar_sensor_to_ego_pose_matrix: npt.NDArray[np.float64]  # (4, 4)
    lidar_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64]  # (4, 4)
    boxes_3d: Sequence[Box3D]


class T4SampleRecordLidarSweepInfo(BaseModel):
    """
    Multisweep lidar information of a T4 sample record, where all attributes
    must be of the same length.

    Attributes:
      lidar_sweep_frame_ids: Lidar sweep frame IDs.
      lidar_sweep_timestamps_seconds: Lidar sweep timestamps in seconds.
      lidar_sweep_pointclouds_paths: Lidar sweep pointclouds paths.
      lidar_sweep_ego_to_global_matrices: Lidar sweep ego to global matrices.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    lidar_sweep_frame_ids: Sequence[str]
    lidar_sweep_timestamps_seconds: Sequence[float]
    lidar_sweep_pointclouds_paths: Sequence[str]
    lidar_sweep_frame_ego_to_global_matrices: Sequence[npt.NDArray[np.float64]]  # (4, 4)
    lidar_sensor_to_lidar_sweep_matrices: Sequence[npt.NDArray[np.float64]]  # (4, 4)

    def model_post_init(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""

        assert (
            len(self.lidar_sweep_frame_ids)
            == len(self.lidar_sweep_timestamps_seconds)
            == len(self.lidar_sweep_pointclouds_paths)
            == len(self.lidar_sweep_frame_ego_to_global_matrices)
            == len(self.lidar_sensor_to_lidar_sweep_matrices)
        ), "All attributes must be of the same length"


class T4SampleRecord(BaseModel):
    """Temporary T4 sample record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    basic_info: T4SampleRecordBasicInfo
    lidar_info: T4SampleRecordLidarInfo
    lidar_sweep_info: T4SampleRecordLidarSweepInfo

    @staticmethod
    def parse_lidar_pointcloud_path(lidar_pointcloud_path: str) -> str:
        """
        Parse lidar pointcloud path to {database_version}/{scene_id}/
        {dataset_version}/data/{lidar_token}/{frame}.bin from path.

        Args:
          lidar_pointcloud_path: Lidar pointcloud path.
        Returns:
          str: Parsed lidar pointcloud path.
        """

        return "/".join(lidar_pointcloud_path.split("/")[-6:])

    def to_dataset_record(self) -> DatasetRecord:
        """
        Convert T4 sample record to dataset record.

        Returns:
          DatasetRecord: Dataset record.
        """

        # Parse the lidar sweep information
        lidar_sweep_pointclouds_paths = [
            self.parse_lidar_pointcloud_path(path)
            for path in self.lidar_sweep_info.lidar_sweep_pointclouds_paths
        ]
        lidar_sweep_ego_to_global_matrices = [
            matrix.astype(np.float32)
            for matrix in self.lidar_sweep_info.lidar_sweep_ego_to_global_matrices
        ]
        lidar_sweep_frame_ego_pose_to_global_matrices = [
            matrix.astype(np.float32)
            for matrix in self.lidar_sweep_info.lidar_sweep_frame_ego_pose_to_global_matrices
        ]
        return DatasetRecord(
            # Basic Metadata
            scenario_id=self.basic_info.scenario_id,
            sample_id=self.basic_info.sample_id,
            sample_index=self.basic_info.sample_index,
            timestamp_seconds=self.basic_info.timestamp_seconds,
            scenario_name=self.basic_info.scenario_name,
            location=self.basic_info.location,
            vehicle_type=self.basic_info.vehicle_type,
            # LiDAR Metadata
            lidar_frame_id=self.lidar_info.lidar_frame_id,
            lidar_sensor_id=self.lidar_info.lidar_sensor_id,
            lidar_sensor_channel_name=self.lidar_info.lidar_sensor_channel_name,
            lidar_pointcloud_path=self.parse_lidar_pointcloud_path(
                self.lidar_info.lidar_pointcloud_path
            ),
            lidar_pointcloud_num_features=self.lidar_info.lidar_pointcloud_num_features,
            lidar_sensor_to_ego_pose_matrix=self.lidar_info.lidar_sensor_to_ego_pose_matrix.astype(
                np.float32
            ),
            lidar_frame_ego_pose_to_global_matrix=self.lidar_info.lidar_frame_ego_pose_to_global_matrix.astype(
                np.float32
            ),
            # Multisweep LiDAR Metadata
            lidar_sweep_frame_ids=self.lidar_sweep_info.lidar_sweep_frame_ids,
            lidar_sweep_timestamps_seconds=self.lidar_sweep_info.lidar_sweep_timestamps_seconds,
            lidar_sweep_pointclouds_paths=lidar_sweep_pointclouds_paths,
            lidar_sweep_ego_to_global_matrices=lidar_sweep_ego_to_global_matrices,
            lidar_sweep_frame_ego_pose_to_global_matrices=lidar_sweep_frame_ego_pose_to_global_matrices,
        )
