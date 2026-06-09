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

from typing import Sequence

from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.frame_basic_metadata import FrameBasicMetadata
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.schemas.lidar_sources import LidarSourceDataModel
from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel


class T4SampleRecord(BaseModel):
    """Temporary T4 sample record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_basic_metadata: FrameBasicMetadata
    lidar_frame_data_models: Sequence[LidarFrameDataModel]
    lidar_source_data_models: Sequence[LidarSourceDataModel]
    category_mapping_data_model: CategoryMappingDataModel

    def to_dataset_record(self) -> DatasetRecord:
        """
        Convert this T4SampleRecord to DatasetRecord.

        Returns:
          DatasetRecord: Dataset record.
        """

        return DatasetRecord(
            scenario_id=self.frame_basic_metadata.scenario_id,
            sample_id=self.frame_basic_metadata.sample_id,
            sample_index=self.frame_basic_metadata.sample_index,
            timestamp_seconds=self.frame_basic_metadata.timestamp_seconds,
            scenario_name=self.frame_basic_metadata.scenario_name,
            location=self.frame_basic_metadata.location,
            vehicle_type=self.frame_basic_metadata.vehicle_type,
            lidar_frames=self.lidar_frame_data_models,
            lidar_sources=self.lidar_source_data_models,
            category_mapping=self.category_mapping_data_model,
        )
