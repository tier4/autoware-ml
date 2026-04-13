import logging

from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict
from t4_devkit import Tier4
from t4_devkit.schema import Sample, SampleData, CalibratedSensor
from t4_devkit.typing import Quaternion, Vector3

from autoware_ml.common.enums.enums import LidarChannel
from autoware_ml.databases.schemas import DatasetRecord
from autoware_ml.databases.scenarios import ScenarioData

logger = logging.getLogger(__name__)


class T4SampleRecord(BaseModel):
    """
    Temporary T4 sample record.

    Attributes:
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      lidar_path: Lidar path.
      location: Location.
      vehicle_type: Vehicle type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    lidar_path: str
    location: str | None
    vehicle_type: str | None

    # Lidar to ego transformation
    lidar2ego_translation: Vector3
    lidar2ego_rotation: Quaternion

    def to_dataset_record(self) -> DatasetRecord:
        """
        Convert T4 sample record to dataset record.

        Returns:
          DatasetRecord: Dataset record.
        """
        return DatasetRecord(
            scenario_id=self.scenario_id,
            sample_id=self.sample_id,
            sample_index=self.sample_index,
            location=self.location,
            vehicle_type=self.vehicle_type,
        )


class T4RecordsGenerator:
    """RecordsGenerator for T4Dataset."""

    def __init__(
        self,
        database_root_path: str,
        scenario_data: ScenarioData,
        max_sweeps: int,
        sample_steps: int,
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
        """

        self.database_root_path = Path(database_root_path)
        self.scenario_data = scenario_data
        self.max_sweeps = max_sweeps
        self.sample_steps = sample_steps
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

    def extract_t4_sample_record(self, sample: Sample, sample_index: int) -> T4SampleRecord:
        """
        Extract T4 sample record from a T4Dataset.

        Args:
          sample: Sample.
          sample_index: Sample index.
        Returns:
          T4SampleRecord: T4 sample record.
        """

        # First, read lidar token from the sample data
        if LidarChannel.LIDAR_TOP in sample.data:
            lidar_token = sample.data[LidarChannel.LIDAR_TOP]
        elif LidarChannel.LIDAR_CONCAT in sample.data:
            lidar_token = sample.data[LidarChannel.LIDAR_CONCAT]
        else:
            raise ValueError(
                f"Lidar channel {LidarChannel.LIDAR_TOP} or {LidarChannel.LIDAR_CONCAT} not found in sample data."
            )

        # Second, read sample data and calibrated sensor from the T4Dataset
        sd_record: SampleData = self.t4_devkit_dataset.get("sample_data", lidar_token)
        cs_record: CalibratedSensor = self.t4_devkit_dataset.get(
            "calibrated_sensor", sd_record.calibrated_sensor_token
        )
        lidar_path, _, _ = self.t4_devkit_dataset.get_sample_data(lidar_token)
        # TODO (KokSeang): Extract more information, for example, boxes and lidar sweeps, from the T4Dataset.
        # Last, return the T4 sample record

        return T4SampleRecord(
            scenario_id=self.scenario_data.scenario_id,
            sample_id=sample.token,
            sample_index=sample_index,
            location=self.scenario_data.location,
            vehicle_type=self.scenario_data.vehicle_type,
            lidar_path=lidar_path,
            lidar2ego_translation=cs_record.translation,
            lidar2ego_rotation=cs_record.rotation,
        )
