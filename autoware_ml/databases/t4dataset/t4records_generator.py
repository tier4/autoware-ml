import logging

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
from t4_devkit import Tier4
from t4_devkit.schema import (
    CalibratedSensor,
    EgoPose,
    LidarSeg,
    Sample,
    SampleData,
    Scene,
    Sensor,
    SchemaName,
)
from t4_devkit.common.timestamp import microseconds2seconds

from autoware_ml.common.enums.enums import LidarChannel, Modality
from autoware_ml.databases.schemas import DatasetRecord
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.t4dataset.t4sample_records import (
    T4SampleRecord,
    BasicMetaData,
    CategoryMetaData,
    LidarMetaData,
    LidarSegMetaData,
    LidarSweepMetaData,
    LidarSourceMetaData,
)
from autoware_ml.utils.dataset import convert_quaternion_to_matrix

logger = logging.getLogger(__name__)


class T4RecordsGenerator:
    """RecordsGenerator for T4Dataset."""

    __MODALITY_STRING = "modality"
    __VALUE_STRING = "value"

    def __init__(
        self,
        database_root_path: str,
        scenario_data: ScenarioData,
        max_sweeps: int,
        sample_steps: int,
        lidar_pointcloud_num_features: int,
    ) -> None:
        """
        Initialize T4RecordsGenerator.

        Args:
          database_root_path: Root path of the T4 database.
          scenario_data: Scenario data.
          max_sweeps: Max number of lidar sweeps to include, only for 3D, set to 0
            if skipping lidar sweep concatenation.
          sample_steps: Number of frames/samples to skip between each sample, set to 1
            if not skipping any samples/frames.
          lidar_pointcloud_num_features: Number of features of the lidar pointcloud.
        """

        self.database_root_path = Path(database_root_path)
        self.scenario_data = scenario_data
        self.max_sweeps = max_sweeps
        self.sample_steps = sample_steps
        self.lidar_pointcloud_num_features = lidar_pointcloud_num_features
        self.t4_devkit_dataset = self._construct_t4_devkit_dataset()

        assert sample_steps > 0, "Sample steps must be greater than 0."
        assert max_sweeps >= 0, "Max sweeps must be greater than or equal to 0."

    def _construct_t4_devkit_dataset(self) -> Tier4:
        """
        Construct T4Devkit class instance.

        Returns:
          Tier4: T4 dataset.
        """

        scene_root_dir_path = (
            self.database_root_path
            / self.scenario_data.dataset_name
            / self.scenario_data.scenario_id
            / self.scenario_data.scenario_version
        )
        if not scene_root_dir_path.exists():
            raise ValueError(f"Scene root directory {scene_root_dir_path} does not exist.")
        return Tier4(data_root=scene_root_dir_path, verbose=False)

    def generate_dataset_records(self) -> Sequence[DatasetRecord]:
        """
        Generate dataset records.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        records = []
        logger.info(
            f"Generating dataset records for scenario: {self.scenario_data.scenario_id} with sample steps: {self.sample_steps} and max sweeps: {self.max_sweeps}"
        )
        for sample_index in range(0, len(self.t4_devkit_dataset.sample), self.sample_steps):
            sample = self.t4_devkit_dataset.sample[sample_index]
            t4_sample_record = self.extract_t4_sample_record(sample, sample_index)
            records.append(t4_sample_record.to_dataset_record())

        return records

    def _extract_basic_metadata(self, sample: Sample, sample_index: int) -> BasicMetaData:
        """
        Extract basic metadata from a T4 sample.

        Args:
          sample: T4 Sample.
          sample_index: Sample index.

        Returns:
          BasicMetaData: Basic metadata of the T4 sample.
        """

        scene_record: Scene = self.t4_devkit_dataset.get(SchemaName.SCENE, sample.scene_token)
        return BasicMetaData(
            scenario_id=self.scenario_data.scenario_id,
            sample_id=sample.token,
            sample_index=sample_index,
            location=self.scenario_data.location,
            vehicle_type=self.scenario_data.vehicle_type,
            timestamp_seconds=microseconds2seconds(sample.timestamp),
            scenario_name=scene_record.name,
        )

    def _extract_lidar_metadata(self, sample: Sample) -> LidarMetaData:
        """
        Extract lidar metadata from a T4 sample.

        Args:
          sample: T4 Sample.

        Returns:
          LidarMetaData: Lidar metadata of the T4 sample.
        """

        # Read lidar channel name
        if LidarChannel.LIDAR_TOP in sample.data:
            lidar_channel_name = LidarChannel.LIDAR_TOP
        elif LidarChannel.LIDAR_CONCAT in sample.data:
            lidar_channel_name = LidarChannel.LIDAR_CONCAT
        else:
            raise ValueError(
                f"Lidar channel {LidarChannel.LIDAR_TOP} or {LidarChannel.LIDAR_CONCAT} not found in sample data."
            )

        calibrated_lidar_sample_data_token = sample.data[lidar_channel_name]
        sd_record: SampleData = self.t4_devkit_dataset.get(
            SchemaName.SAMPLE_DATA, calibrated_lidar_sample_data_token
        )
        cs_record: CalibratedSensor = self.t4_devkit_dataset.get(
            SchemaName.CALIBRATED_SENSOR, sd_record.calibrated_sensor_token
        )
        lidar_sensor_to_ego_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=cs_record.rotation,
            translation=cs_record.translation,
            convert_to_float32=False,
        )

        lidar_path, boxes_3d, _ = self.t4_devkit_dataset.get_sample_data(
            sample_data_token=calibrated_lidar_sample_data_token,
            as_3d=True,
            as_sensor_coord=True,
        )

        # Extract ego pose to global matrix in the lidar frame from the T4Dataset
        ego_pose_record: EgoPose = self.t4_devkit_dataset.get(
            SchemaName.EGO_POSE, sd_record.ego_pose_token
        )
        lidar_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=ego_pose_record.rotation,
            translation=ego_pose_record.translation,
            convert_to_float32=False,
        )

        return LidarMetaData(
            lidar_frame_id=calibrated_lidar_sample_data_token,
            lidar_sensor_id=cs_record.token,
            lidar_sensor_channel_name=lidar_channel_name,
            lidar_pointcloud_path=lidar_path,
            lidar_pointcloud_source_path=sd_record.info_filename,
            lidar_pointcloud_num_features=self.lidar_pointcloud_num_features,
            lidar_sensor_to_ego_pose_matrix=lidar_sensor_to_ego_matrix,
            lidar_frame_ego_pose_to_global_matrix=lidar_frame_ego_pose_to_global_matrix,
            boxes_3d=boxes_3d,
        )

    def _compute_sensor_transformation_matrices(
        self,
        sensor_sample_data_record: SampleData,
        selected_sensor_to_ego_pose_matrix: npt.NDArray[np.float64],
        selected_sensor_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute transformation matrices for a sensor.

        Args:
            sensor_sample_data_record: Sample data record of the sensor.
            selected_sensor_to_ego_pose_matrix: Transformation matrix from the selected
              sensor to its' the ego pose.
            selected_sensor_frame_ego_pose_to_global_matrix: Transformation matrix from the selected
              sensor frame ego pose to the global frame.

        Returns:
            Tuple of transformation matrices:
              1. Sensor frame ego pose to global matrix (4x4)
              2. Selected sensor to sensor transformation matrix (4x4)
        """
        sensor_calibrated_sensor_record: CalibratedSensor = self.t4_devkit_dataset.get(
            SchemaName.CALIBRATED_SENSOR, sensor_sample_data_record.calibrated_sensor_token
        )
        sensor_ego_pose_record: EgoPose = self.t4_devkit_dataset.get(
            SchemaName.EGO_POSE, sensor_sample_data_record.ego_pose_token
        )

        sensor_to_ego_pose_tranlation = sensor_calibrated_sensor_record.translation
        sensor_to_ego_pose_rotation = sensor_calibrated_sensor_record.rotation

        sensor_frame_ego_pose_to_global_translation = sensor_ego_pose_record.translation
        sensor_frame_ego_pose_to_global_rotation = sensor_ego_pose_record.rotation

        sensor_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=sensor_frame_ego_pose_to_global_rotation,
            translation=sensor_frame_ego_pose_to_global_translation,
            convert_to_float32=False,
        )

        sensor_to_ego_pose_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=sensor_to_ego_pose_rotation,
            translation=sensor_to_ego_pose_tranlation,
            convert_to_float32=False,
        )

        # Compute the transformation matrix of sensor to the selected sensor coordinate
        # Sensor -> sensor frame ego pose -> global -> selected sensor frame ego pose -> selected sensor
        # For example, if the sensor is a lidar sweep, and the selected sensor is the top lidar sweep:
        # Sweep -> sweep frame ego pose -> global -> top lidar frame ego pose -> top lidar
        # Right-to-left multiplication:
        sensor_to_selected_sensor_matrix = (
            np.linalg.inv(selected_sensor_to_ego_pose_matrix)
            @ np.linalg.inv(selected_sensor_frame_ego_pose_to_global_matrix)
            @ sensor_frame_ego_pose_to_global_matrix
            @ sensor_to_ego_pose_matrix
        )
        return sensor_frame_ego_pose_to_global_matrix, sensor_to_selected_sensor_matrix

    def _extract_lidar_sweep_metadata(
        self, t4_sample_record_lidar_info: LidarMetaData
    ) -> LidarSweepMetaData:
        """
        Extract multisweep lidar metadata from a T4 Sample.

        Args:
            t4_sample_record_lidar_info: T4 Sample lidar metadata.

        Returns:
            LidarSweepsMetaData: T4 sample lidar sweep metadata
            corresponding to the current T4 sample.
        """

        current_lidar_sample_data_token = t4_sample_record_lidar_info.lidar_frame_id
        lidar_sweep_frame_ids = []
        lidar_sweep_timestamps_seconds = []
        lidar_sweep_pointclouds_paths = []
        lidar_sweep_frame_ego_pose_to_global_matrices = []
        lidar_sensor_to_lidar_sweep_matrices = []

        current_sample_data_record: SampleData = self.t4_devkit_dataset.get(
            SchemaName.SAMPLE_DATA, current_lidar_sample_data_token
        )

        for _ in range(self.max_sweeps):
            # Stop processing if the current lidar sample data has no previous sample data
            if not current_sample_data_record.prev:
                break

            current_sample_data_record: SampleData = self.t4_devkit_dataset.get(
                SchemaName.SAMPLE_DATA, current_sample_data_record.prev
            )
            lidar_sweep_frame_ids.append(current_sample_data_record.token)
            lidar_sweep_timestamps_seconds.append(
                microseconds2seconds(current_sample_data_record.timestamp)
            )
            lidar_sweep_pointclouds_paths.append(
                self.t4_devkit_dataset.get_sample_data_path(
                    sample_data_token=current_sample_data_record.token
                )
            )

            # Get the current lidar sweep frame ego pose
            lidar_sweep_transformations = self._compute_sensor_transformation_matrices(
                sensor_sample_data_record=current_sample_data_record,
                selected_sensor_to_ego_pose_matrix=t4_sample_record_lidar_info.lidar_sensor_to_ego_pose_matrix,
                selected_sensor_frame_ego_pose_to_global_matrix=t4_sample_record_lidar_info.lidar_frame_ego_pose_to_global_matrix,
            )
            lidar_sweep_frame_ego_pose_to_global_matrix, lidar_sweep_to_lidar_sensor_matrix = (
                lidar_sweep_transformations
            )

            # Inverse it to obtain the transformation matrix
            # from the lidar sensor to the lidar sweeps
            lidar_sensor_to_lidar_sweep_matrix = np.linalg.inv(lidar_sweep_to_lidar_sensor_matrix)
            lidar_sweep_frame_ego_pose_to_global_matrices.append(
                lidar_sweep_frame_ego_pose_to_global_matrix
            )
            lidar_sensor_to_lidar_sweep_matrices.append(lidar_sensor_to_lidar_sweep_matrix)

        return LidarSweepMetaData(
            lidar_sweep_frame_ids=lidar_sweep_frame_ids,
            lidar_sweep_timestamps_seconds=lidar_sweep_timestamps_seconds,
            lidar_sweep_pointclouds_paths=lidar_sweep_pointclouds_paths,
            lidar_sweep_frame_ego_pose_to_global_matrices=lidar_sweep_frame_ego_pose_to_global_matrices,
            lidar_sensor_to_lidar_sweep_matrices=lidar_sensor_to_lidar_sweep_matrices,
        )

    def _extract_lidar_source_metadata(self) -> LidarSourceMetaData:
        """
        Extract lidar sources metadata from a T4 Sample.

        Args:
          sample: T4 Sample.

        Returns:
          LidarSourcesMetaData: Lidar sources metadata of the T4 sample.
        """

        # First, read lidar source sensor tokens from the sample data
        calibrated_sensor_records: Sequence[CalibratedSensor] = getattr(
            self.t4_devkit_dataset, SchemaName.CALIBRATED_SENSOR, []
        )

        if not len(calibrated_sensor_records):
            return LidarSourceMetaData(
                lidar_source_sensor_tokens=[],
                lidar_source_translations=[],
                lidar_source_rotations=[],
            )

        lidar_source_channel_names = []
        lidar_source_sensor_tokens = []
        lidar_source_translations = []
        lidar_source_rotations = []
        for calibrated_sensor_record in calibrated_sensor_records:
            try:
                sensor_record: Sensor = self.t4_devkit_dataset.get(
                    SchemaName.SENSOR, calibrated_sensor_record.sensor_token
                )
            except ValueError:
                continue

            modality = getattr(sensor_record, self.__MODALITY_STRING, None)
            modality_value = getattr(modality, self.__VALUE_STRING, None)
            if modality_value != Modality.LIDAR:
                continue

            if sensor_record.channel not in lidar_source_channel_names:
                lidar_source_channel_names.append(sensor_record.channel)
                lidar_source_sensor_tokens.append(sensor_record.token)
                lidar_source_translations.append(calibrated_sensor_record.translation)
                lidar_source_rotations.append(calibrated_sensor_record.rotation.rotation_matrix)

        return LidarSourceMetaData(
            lidar_source_channel_names=lidar_source_channel_names,
            lidar_source_sensor_tokens=lidar_source_sensor_tokens,
            lidar_source_translations=lidar_source_translations,
            lidar_source_rotations=lidar_source_rotations,
        )

    def _extract_lidarseg_metadata(
        self, sample_index: int, calibrated_lidar_sample_data_token: str
    ) -> LidarSegMetaData:
        """
        Extract lidarseg metadata from a T4 Sample.

        Args:
          sample_index: Sample index.
          calibrated_lidar_sample_data_token: Calibrated lidar sample data token.

        Returns:
          LidarSegMetaData: Lidarseg metadata of the T4 sample.
        """
        lidarseg_records: Sequence[LidarSeg] = getattr(
            self.t4_devkit_dataset, SchemaName.LIDARSEG, []
        )
        if not len(lidarseg_records):
            return LidarSegMetaData(
                lidarseg_pts_semantic_mask_path=None,
                lidarseg_pts_semantic_mask_category_names=[],
                lidarseg_pts_semantic_mask_category_indices=[],
            )

        assert sample_index < len(lidarseg_records), (
            "Sample index is out of range of lidarseg records."
        )

        current_lidarseg_record = lidarseg_records[sample_index]
        assert current_lidarseg_record.sample_data_token == calibrated_lidar_sample_data_token, (
            "Lidarseg record sample data token does not match the calibrated lidar sample data token."
        )

        return LidarSegMetaData(
            lidarseg_pts_semantic_mask_path=current_lidarseg_record.filename,
        )

    def _extract_category_metadata(self) -> CategoryMetaData:
        """
        Extract category metadata from a T4 Sample.

        Args:
          sample_index: Sample index.

        Returns:
          CategoryMetaData: Category metadata of the T4 sample.
        """

        category_records = self.t4_devkit_dataset.get_table(SchemaName.CATEGORY)
        if not len(category_records):
            return CategoryMetaData(
                category_names=[],
                category_indices=[],
            )

        category_names = []
        category_indices = []
        for category_record in category_records:
            category_names.append(category_record.name)
            category_indices.append(category_record.index)

        return CategoryMetaData(
            category_names=category_names,
            category_indices=category_indices,
        )

    def extract_t4_sample_record(self, sample: Sample, sample_index: int) -> T4SampleRecord:
        """
        Extract T4 sample record from a T4Dataset.

        Args:
          sample: Sample.
          sample_index: Sample index.
        Returns:
          T4SampleRecord: T4 sample record.
        """

        # 1) Extract basic information from the T4Dataset
        basic_metadata = self._extract_basic_metadata(sample=sample, sample_index=sample_index)

        # 2) Extract lidar information from the T4Dataset
        lidar_metadata = self._extract_lidar_metadata(sample=sample)

        # 3) Extract multisweep lidar information from the T4Dataset
        lidar_sweep_metadata = self._extract_lidar_sweep_metadata(
            t4_sample_record_lidar_info=lidar_metadata
        )

        # 4) Extract lidar sources information from the T4Dataset
        lidar_source_metadata = self._extract_lidar_source_metadata()

        # 5) Extract lidarseg information from the T4Dataset
        lidarseg_metadata = self._extract_lidarseg_metadata(
            sample_index=sample_index,
            calibrated_lidar_sample_data_token=lidar_metadata.lidar_frame_id,
        )

        # 6) Extract category information from the T4Dataset
        category_metadata = self._extract_category_metadata()

        return T4SampleRecord(
            basic_metadata=basic_metadata,
            lidar_metadata=lidar_metadata,
            lidar_sweep_metadata=lidar_sweep_metadata,
            lidar_source_metadata=lidar_source_metadata,
            lidarseg_metadata=lidarseg_metadata,
            category_metadata=category_metadata,
        )
