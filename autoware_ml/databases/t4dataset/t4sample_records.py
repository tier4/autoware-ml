import logging

from typing import Sequence, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from t4_devkit.dataclass.box import Box3D

from autoware_ml.databases.schemas import DatasetRecord

logger = logging.getLogger(__name__)


class BasicMetaData(BaseModel):
    """
    Basic metadata of a T4 sample record.

    Attributes:
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      timestamp_seconds: Timestamp in seconds.
      scenario_name: Scenario name.
      location: Location.
      vehicle_type: Vehicle type.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    scenario_name: str
    location: str | None
    vehicle_type: str | None


class LidarMetaData(BaseModel):
    """
    Lidar metadata of a T4 sample record.

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
    lidar_pointcloud_source_path: str | None
    lidar_pointcloud_num_features: int
    lidar_sensor_to_ego_pose_matrix: npt.NDArray[np.float64]  # (4, 4)
    lidar_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64]  # (4, 4)
    boxes_3d: Sequence[Box3D]

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


class LidarSweepMetaData(BaseModel):
    """
    Multisweep lidar metadata of a T4 sample record, where all attributes
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
    lidar_sweep_frame_ego_pose_to_global_matrices: Sequence[npt.NDArray[np.float64]]  # (4, 4)
    lidar_sensor_to_lidar_sweep_matrices: Sequence[npt.NDArray[np.float64]]  # (4, 4)

    def model_post_init(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""

        assert (
            len(self.lidar_sweep_frame_ids)
            == len(self.lidar_sweep_timestamps_seconds)
            == len(self.lidar_sweep_pointclouds_paths)
            == len(self.lidar_sweep_frame_ego_pose_to_global_matrices)
            == len(self.lidar_sensor_to_lidar_sweep_matrices)
        ), "All attributes must be of the same length"

    @property
    def lidar_sweep_pointclouds_relative_paths(self: str) -> Sequence[str]:
        """
        Parse lidar sweep pointclouds paths to {database_version}/{scene_id}/
        {dataset_version}/data/{lidar_token}/{frame}.bin from path.

        Returns:
          Sequence[str]: List of lidar sweep pointclouds relative paths.
        """

        return [
            "/".join(sweep_pointclouds_path.split("/")[-6:])
            for sweep_pointclouds_path in self.lidar_sweep_pointclouds_paths
        ]

    @property
    def lidar_sweep_frame_ego_pose_to_global_matrices_fp32(
        self,
    ) -> Sequence[npt.NDArray[np.float32]]:
        """
        Convert the lidar sweep frame ego pose to global matrices to float32.

        Returns:
          Sequence[npt.NDArray[np.float32]]: Lidar sweep frame ego pose to global matrices.
        """

        return [
            matrix.astype(np.float32)
            for matrix in self.lidar_sweep_frame_ego_pose_to_global_matrices
        ]

    @property
    def lidar_sensor_to_lidar_sweep_matrices_fp32(self) -> Sequence[npt.NDArray[np.float32]]:
        """
        Convert the lidar sensor to lidar sweep matrices to float32.

        Returns:
          Sequence[npt.NDArray[np.float32]]: Lidar sensor to lidar sweep matrices.
        """

        return [matrix.astype(np.float32) for matrix in self.lidar_sensor_to_lidar_sweep_matrices]


class LidarSourceMetaData(BaseModel):
    """
    Lidar source metadata of a T4 sample record.

    Attributes:
      lidar_source_channel_names: List of Lidar source channel names.
      lidar_source_sensor_tokens: List of Lidar source sensor tokens.
      lidar_source_translations: List of Lidar source translations.
      lidar_source_rotations: List of Lidar source rotations.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    lidar_source_channel_names: Sequence[str]
    lidar_source_sensor_tokens: Sequence[str]
    lidar_source_translations: Sequence[npt.NDArray[np.float64]]
    lidar_source_rotations: Sequence[npt.NDArray[np.float64]]

    def model_post_init(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""

        assert (
            len(self.lidar_source_channel_names)
            == len(self.lidar_source_sensor_tokens)
            == len(self.lidar_source_translations)
            == len(self.lidar_source_rotations)
        ), "All attributes must be of the same length"

    @property
    def lidar_source_translations_fp32(self) -> Sequence[npt.NDArray[np.float32]]:
        """
        Convert the lidar source translations to float32.

        Returns:
          Sequence[npt.NDArray[np.float32]]: Lidar source translations.
        """

        return [translation.astype(np.float32) for translation in self.lidar_source_translations]

    @property
    def lidar_source_rotations_fp32(self) -> Sequence[npt.NDArray[np.float32]]:
        """
        Convert the lidar source rotations to float32.

        Returns:
          Sequence[npt.NDArray[np.float32]]: Lidar source rotations.
        """

        return [rotation.astype(np.float32) for rotation in self.lidar_source_rotations]


class CategoryMetaData(BaseModel):
    """
    Category metadata of a T4 sample record.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    category_names: Sequence[str]
    category_indices: Sequence[int]

    def model_post_init(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""

        assert len(self.category_names) == len(self.category_indices), (
            "All attributes must be of the same length"
        )


class Boxes3DMetadata(BaseModel):
    """
    Metadata of 3D bounding boxes.

    Attributes:
      boxes_3d_arrays: Sequence of 3D bounding boxes arrays.
      boxes_3d_instance_ids: Sequence of instance IDs for each 3D bounding box.
      boxes_3d_dataset_label_names: Sequence of dataset label names for each 3D bounding box.
      boxes_3d_label_names: Sequence of label names for each 3D bounding box.
      boxes_3d_label_indices: Sequence of label indices for each 3D bounding box.
      boxes_3d_num_lidar_pointclouds: Sequence of number of lidar pointclouds for each 3D bounding box.
      boxes_3d_num_radar_pointclouds: Sequence of number of radar pointclouds for each 3D bounding box.
      boxes_3d_valid: Sequence of valid flags for each 3D bounding box.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    boxes_3d_arrays: Sequence[npt.NDArray[np.float32]]
    boxes_3d_instance_ids: Sequence[str]
    boxes_3d_dataset_label_names: Sequence[str]
    boxes_3d_label_names: Sequence[str]
    boxes_3d_label_indices: Sequence[int]
    boxes_3d_num_lidar_pointclouds: Sequence[int]
    boxes_3d_num_radar_pointclouds: Sequence[int]
    boxes_3d_valid: Sequence[bool]
    boxes_3d_attributes: Sequence[set[str]]

    def __model_post_init__(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""
        assert (
            len(self.indices)
            == len(self.boxes_3d)
            == len(self.boxes_3d_instance_ids)
            == len(self.boxes_3d_dataset_names)
            == len(self.boxes_3d_label_names)
            == len(self.boxes_3d_label_indices)
            == len(self.boxes_3d_num_lidar_pointclouds)
            == len(self.boxes_3d_num_radar_pointclouds)
            == len(self.boxes_3d_valid)
            == len(self.boxes_3d_attributes)
        ), "All attributes must be of the same length"


class LidarSegMetaData(BaseModel):
    """
    Lidarseg metadata of a T4 sample record.

    Attributes:
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    lidarseg_pts_semantic_mask_path: str | None

    @property
    def lidarseg_pts_semantic_mask_relative_path(self: str) -> str | None:
        """
        Parse lidarseg pts semantic mask path to {database_version}/{scene_id}/
        {dataset_version}/data/{lidar_token}/{frame}.bin from path.
        """
        if self.lidarseg_pts_semantic_mask_path is None:
            return None

        return "/".join(self.lidarseg_pts_semantic_mask_path.split("/")[-6:])


class T4SampleRecord(BaseModel):
    """Temporary T4 sample record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    basic_metadata: BasicMetaData
    lidar_metadata: LidarMetaData
    lidar_sweep_metadata: LidarSweepMetaData
    lidar_source_metadata: LidarSourceMetaData
    lidarseg_metadata: LidarSegMetaData
    category_metadata: CategoryMetaData
    boxes_3d_metadata: Boxes3DMetadata

    def to_dataset_record(self) -> DatasetRecord:
        """
        Convert this T4SampleRecord to DatasetRecord.

        Returns:
          DatasetRecord: Dataset record.
        """

        return DatasetRecord(
            # Basic Metadata
            scenario_id=self.basic_metadata.scenario_id,
            sample_id=self.basic_metadata.sample_id,
            sample_index=self.basic_metadata.sample_index,
            timestamp_seconds=self.basic_metadata.timestamp_seconds,
            scenario_name=self.basic_metadata.scenario_name,
            location=self.basic_metadata.location,
            vehicle_type=self.basic_metadata.vehicle_type,
            # LiDAR Metadata
            lidar_frame_id=self.lidar_metadata.lidar_frame_id,
            lidar_sensor_id=self.lidar_metadata.lidar_sensor_id,
            lidar_sensor_channel_name=self.lidar_metadata.lidar_sensor_channel_name,
            lidar_pointcloud_path=self.lidar_metadata.lidar_pointcloud_relative_path,
            lidar_pointcloud_source_path=self.lidar_metadata.lidar_pointcloud_source_relative_path,
            lidar_pointcloud_num_features=self.lidar_metadata.lidar_pointcloud_num_features,
            lidar_sensor_to_ego_pose_matrix=self.lidar_metadata.lidar_sensor_to_ego_pose_matrix_fp32,
            lidar_frame_ego_pose_to_global_matrix=self.lidar_metadata.lidar_frame_ego_pose_to_global_matrix_fp32,
            # Multisweep LiDAR Metadata
            lidar_sweep_frame_ids=self.lidar_sweep_metadata.lidar_sweep_frame_ids,
            lidar_sweep_timestamps_seconds=self.lidar_sweep_metadata.lidar_sweep_timestamps_seconds,
            lidar_sweep_pointclouds_paths=self.lidar_sweep_metadata.lidar_sweep_pointclouds_relative_paths,
            lidar_sweep_frame_ego_pose_to_global_matrices=self.lidar_sweep_metadata.lidar_sweep_frame_ego_pose_to_global_matrices_fp32,
            lidar_sensor_to_lidar_sweep_matrices=self.lidar_sweep_metadata.lidar_sensor_to_lidar_sweep_matrices_fp32,
            # Lidar sources Metadata
            lidar_source_channel_names=self.lidar_source_metadata.lidar_source_channel_names,
            lidar_source_sensor_tokens=self.lidar_source_metadata.lidar_source_sensor_tokens,
            lidar_source_translations=self.lidar_source_metadata.lidar_source_translations_fp32,
            lidar_source_rotations=self.lidar_source_metadata.lidar_source_rotations_fp32,
            # Lidarseg Metadata
            lidarseg_pts_semantic_mask_path=self.lidarseg_metadata.lidarseg_pts_semantic_mask_relative_path,
            # Category Metadata
            category_names=self.category_metadata.category_names,
            category_indices=self.category_metadata.category_indices,
            # Boxes 3D Metadata
            boxes_3d_arrays=self.boxes_3d_metadata.boxes_3d_arrays,
            boxes_3d_dataset_label_names=self.boxes_3d_metadata.boxes_3d_dataset_label_names,
            boxes_3d_label_names=self.boxes_3d_metadata.boxes_3d_label_names,
            boxes_3d_label_indices=self.boxes_3d_metadata.boxes_3d_label_indices,
            boxes_3d_instance_ids=self.boxes_3d_metadata.boxes_3d_instance_ids,
            boxes_3d_num_lidar_pointclouds=self.boxes_3d_metadata.boxes_3d_num_lidar_pointclouds,
            boxed_3d_num_radar_pointclouds=self.boxes_3d_metadata.boxes_3d_num_radar_pointclouds,
            boxes_3d_valid=self.boxes_3d_metadata.boxes_3d_valid,
            boxes_3d_attributes=self.boxes_3d_metadata.boxes_3d_attributes,
        )
