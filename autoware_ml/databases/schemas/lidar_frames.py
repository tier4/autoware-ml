from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DatasetTableColumn,
    DataModelInterface,
)


@dataclass(frozen=True)
class LidarFrameDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to lidar.
    """

    lidar_frame_id = DatasetTableColumn("lidar_frame_id", pl.String)
    lidar_keyframe = DatasetTableColumn("lidar_keyframe", pl.Boolean)
    lidar_sensor_id = DatasetTableColumn("lidar_sensor_id", pl.String)
    lidar_timestamp_seconds = DatasetTableColumn("lidar_timestamp_seconds", pl.Float64)
    lidar_sensor_channel_name = DatasetTableColumn("lidar_sensor_channel_name", pl.String)
    lidar_pointcloud_path = DatasetTableColumn("lidar_pointcloud_path", pl.String)
    lidar_pointcloud_source_path = DatasetTableColumn("lidar_pointcloud_source_path", pl.String)
    lidar_pointcloud_num_features = DatasetTableColumn("lidar_pointcloud_num_features", pl.Int32)
    lidar_sensor_to_ego_pose_matrix = DatasetTableColumn(
        "lidar_sensor_to_ego_pose_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )
    lidar_frame_ego_pose_to_global_matrix = DatasetTableColumn(
        "lidar_frame_ego_pose_to_global_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )
    lidar_sensor_to_lidar_sweep_matrices = DatasetTableColumn(
        "lidar_sensor_to_lidar_sweep_matrices", pl.Array(pl.Float32, shape=(4, 4))
    )
    lidar_pointcloud_semantic_mask_path = DatasetTableColumn(
        "lidar_pointcloud_semantic_mask_path", pl.String
    )


class LidarFrameDataModel(BaseModel, DataModelInterface):
    """
    Lidar frame data model that can be shared by multiple datasets. It saves the metadata of a lidar
    frame. Note that lidar sweeps also use this data model.

    Attributes:
      lidar_frame_id: Lidar frame ID.
      lidar_keyframe: Whether this lidar frame is a keyframe. Set to True if it's a keyframe,
        otherwise, it is a sweep frame.
      lidar_sensor_id: Lidar sensor ID.
      lidar_sensor_channel_name: Lidar sensor channel name.
      lidar_timestamp_seconds: Lidar timestamp in seconds.
      lidar_pointcloud_path: Lidar pointcloud path.
      lidar_pointcloud_source_path: Lidar pointcloud source path, which is the path to the
        information for each lidar pointcloud. Set to None if it's not available.
      lidar_pointcloud_num_features: Lidar pointcloud num features.
      lidar_sensor_to_ego_pose_matrix: Transformation matrix from the lidar sensor of this frame to
        the ego pose of this lidar frame.
      lidar_frame_ego_pose_to_global_matrix: Transformation matrix from the ego pose of this lidar
        frame to the global frame.
      lidar_sensor_to_lidar_sweep_matrices: Transformation matrices from the main lidar sensor
        to other lidar sweeps at this frame.
      lidar_pointcloud_semantic_mask_path: Lidar pointcloud semantic mask path. Set to None if it's
        not available.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    lidar_frame_id: str
    lidar_keyframe: bool
    lidar_sensor_id: str
    lidar_sensor_channel_name: str
    lidar_timestamp_seconds: float
    lidar_pointcloud_path: str
    lidar_pointcloud_source_path: str | None
    lidar_pointcloud_num_features: int
    lidar_sensor_to_ego_pose_matrix: npt.NDArray[np.float64]  # (4, 4)
    # Transformation matrix from the ego pose of this lidar frame to the global frame.
    lidar_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64]  # (4, 4)
    # Transformation matrices from the main lidar sensor to other lidar sweeps at this frame.
    lidar_sensor_to_lidar_sweep_matrices: npt.NDArray[np.float64]  # (4, 4)
    lidar_pointcloud_semantic_mask_path: str | None

    @property
    def lidar_pointcloud_relative_path(self: str) -> str:
        """
        Parse lidar pointcloud path to {database_version}/{scene_id}/
        {dataset_version}/data/{lidar_token}/{frame}.bin from path.

        Returns:
          str: Lidar pointcloud relative path.
        """

        return "/".join(self.lidar_pointcloud_path.split("/")[-6:])

    @property
    def lidar_pointcloud_source_relative_path(self: str) -> str | None:
        """
        Parse lidar pointcloud source path to {database_version}/{scene_id}/
        {dataset_version}/data/{lidar_token}/{frame}.bin from path.

        Returns:
          str | None: Lidar pointcloud source relative path.
        """
        if self.lidar_pointcloud_source_path is None:
            return None

        return "/".join(self.lidar_pointcloud_source_path.split("/")[-6:])

    @property
    def lidarseg_pointcloud_semantic_mask_relative_path(self: str) -> str | None:
        """
        Parse lidarseg pts semantic mask path to {database_version}/{scene_id}/
        {dataset_version}/data/{lidar_token}/{frame}.bin from path.
        """
        if self.lidar_pointcloud_semantic_mask_path is None:
            return None

        return "/".join(self.lidar_pointcloud_semantic_mask_path.split("/")[-6:])

    @property
    def lidar_sensor_to_ego_pose_matrix_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the lidar sensor to ego pose matrix to float32.

        Returns:
          npt.NDArray[np.float32]: Lidar sensor to ego pose matrix.
        """

        return self.lidar_sensor_to_ego_pose_matrix.astype(np.float32)

    @property
    def lidar_frame_ego_pose_to_global_matrix_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the lidar frame ego pose to global matrix to float32.

        Returns:
          npt.NDArray[np.float32]: Lidar frame ego pose to global matrix.
        """

        return self.lidar_frame_ego_pose_to_global_matrix.astype(np.float32)

    @property
    def lidar_sensor_to_lidar_sweep_matrices_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the lidar sensor to lidar sweep matrices to float32.

        Returns:
          npt.NDArray[np.float32] | None: Lidar sensor to lidar sweep matrices.
        """

        return self.lidar_sensor_to_lidar_sweep_matrices.astype(np.float32)

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the lidar frame data model to a dictionary.

        Args:
          to_fp32: Whether to convert the lidar frame data model to float32.

        Returns:
          Mapping[str, Any]: Dictionary representation of the lidar frame data model.
        """

        return {
            LidarFrameDatasetSchema.lidar_frame_id.name: self.lidar_frame_id,
            LidarFrameDatasetSchema.lidar_keyframe.name: self.lidar_keyframe,
            LidarFrameDatasetSchema.lidar_sensor_id.name: self.lidar_sensor_id,
            LidarFrameDatasetSchema.lidar_timestamp_seconds.name: self.lidar_timestamp_seconds,
            LidarFrameDatasetSchema.lidar_sensor_channel_name.name: self.lidar_sensor_channel_name,
            LidarFrameDatasetSchema.lidar_pointcloud_path.name: self.lidar_pointcloud_path,
            LidarFrameDatasetSchema.lidar_pointcloud_source_path.name: self.lidar_pointcloud_source_path,
            LidarFrameDatasetSchema.lidar_pointcloud_num_features.name: self.lidar_pointcloud_num_features,
            LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: self.lidar_sensor_to_ego_pose_matrix_fp32,
            LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: self.lidar_frame_ego_pose_to_global_matrix_fp32,
            LidarFrameDatasetSchema.lidar_sensor_to_lidar_sweep_matrices.name: self.lidar_sensor_to_lidar_sweep_matrices_fp32,
            LidarFrameDatasetSchema.lidar_pointcloud_semantic_mask_path.name: self.lidar_pointcloud_semantic_mask_path,
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> LidarFrameDataModel:
        """
        Load the lidar frame data model and decode it to the corresponding LidarFrameDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the lidar frame data model, which is
          deserialized from a Polars dataframe.

        Returns:
          LidarFrameDataModel: LidarFrameDataModel object.
        """

        return cls(
            lidar_frame_id=data_model[LidarFrameDatasetSchema.lidar_frame_id.name],
            lidar_keyframe=data_model[LidarFrameDatasetSchema.lidar_keyframe.name],
            lidar_sensor_id=data_model[LidarFrameDatasetSchema.lidar_sensor_id.name],
            lidar_timestamp_seconds=data_model[
                LidarFrameDatasetSchema.lidar_timestamp_seconds.name
            ],
            lidar_sensor_channel_name=data_model[
                LidarFrameDatasetSchema.lidar_sensor_channel_name.name
            ],
            lidar_pointcloud_path=data_model[LidarFrameDatasetSchema.lidar_pointcloud_path.name],
            lidar_pointcloud_source_path=data_model[
                LidarFrameDatasetSchema.lidar_pointcloud_source_path.name
            ],
            lidar_pointcloud_num_features=data_model[
                LidarFrameDatasetSchema.lidar_pointcloud_num_features.name
            ],
            lidar_sensor_to_ego_pose_matrix=data_model[
                LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name
            ],
            lidar_frame_ego_pose_to_global_matrix=data_model[
                LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name
            ],
            lidar_sensor_to_lidar_sweep_matrices=data_model[
                LidarFrameDatasetSchema.lidar_sensor_to_lidar_sweep_matrices.name
            ],
            lidar_pointcloud_semantic_mask_path=data_model[
                LidarFrameDatasetSchema.lidar_pointcloud_semantic_mask_path.name
            ],
        )
