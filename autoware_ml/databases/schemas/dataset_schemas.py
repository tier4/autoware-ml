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

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Any

import polars as pl
from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.base_schemas import DatasetTableColumn, DataModelInterface
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema, LidarFrameDataModel
from autoware_ml.databases.schemas.category_mapping import (
    CategoryMappingDataModel,
    CategoryMappingDatasetSchema,
)
from autoware_ml.databases.schemas.lidar_sources import (
    LidarSourceDatasetSchema,
    LidarSourceDataModel,
)


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
      LIDAR_FRAMES: Lidar frames colum, which is a list of dictionaries to save metadata of a lidar
        frame. It also saves lidar sweeps as each item here.

      # Lidar Sources Schema
      LIDAR_SOURCES: Lidar sources column, which is a list of dictionaries to save metadata about
        each lidar sensor.

      # Category Schema
      CATEGORY_MAPPING: Category mapping column, which is a dictionary to save the mapping between
        category names and category indices.
    """

    # Basic Schema
    SCENARIO_ID = DatasetTableColumn("scenario_id", pl.String)
    SAMPLE_ID = DatasetTableColumn("sample_id", pl.String)
    SAMPLE_INDEX = DatasetTableColumn("sample_index", pl.Int32)
    TIMESTAMP_SECONDS = DatasetTableColumn("timestamp_seconds", pl.Float64)
    LOCATION = DatasetTableColumn("location", pl.String)
    VEHICLE_TYPE = DatasetTableColumn("vehicle_type", pl.String)
    SCENARIO_NAME = DatasetTableColumn("scenario_name", pl.String)

    # LiDAR Frames Schema
    LIDAR_FRAMES = DatasetTableColumn(
        "lidar_frames", pl.List(pl.Struct(LidarFrameDatasetSchema.to_polars_field_schema()))
    )

    # LiDAR Sources Schema
    LIDAR_SOURCES = DatasetTableColumn(
        "lidar_sources", pl.List(pl.Struct(LidarSourceDatasetSchema.to_polars_field_schema()))
    )

    # Category Schema
    CATEGORY_MAPPING = DatasetTableColumn(
        "category_mapping",
        pl.Struct(CategoryMappingDatasetSchema.to_polars_field_schema()),
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


class DatasetRecord(BaseModel, DataModelInterface):
    """
    Data class to save a record for each column in the annotation table.

    Attributes:
      # Basic Metadata
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      location: Location of the vehicle.
      vehicle_type: Type of the vehicle.

      # LiDAR frame data
      lidar_frames: List of lidar frame data models, including multisweep lidar frames.

      # Lidar sources data
      lidar_sources: List of lidar source data models.

      # Category data
      category_mapping: Category mapping data model.
    """

    # Set model config to frozen
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # Basic Dataset Record
    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    location: str
    vehicle_type: str
    scenario_name: str

    lidar_frames: Sequence[LidarFrameDataModel]
    lidar_sources: Sequence[LidarSourceDataModel] | None
    category_mapping: CategoryMappingDataModel | None

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the dataset record to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the dataset record.
        """
        data_model = {
            DatasetTableSchema.SCENARIO_ID.name: self.scenario_id,
            DatasetTableSchema.SAMPLE_ID.name: self.sample_id,
            DatasetTableSchema.SAMPLE_INDEX.name: self.sample_index,
            DatasetTableSchema.TIMESTAMP_SECONDS.name: self.timestamp_seconds,
            DatasetTableSchema.LOCATION.name: self.location,
            DatasetTableSchema.VEHICLE_TYPE.name: self.vehicle_type,
            DatasetTableSchema.SCENARIO_NAME.name: self.scenario_name,
        }
        data_model[DatasetTableSchema.LIDAR_FRAMES.name] = [
            lidar_frame.to_dictionary() for lidar_frame in self.lidar_frames
        ]

        if self.lidar_sources:
            data_model[DatasetTableSchema.LIDAR_SOURCES.name] = [
                lidar_source.to_dictionary() for lidar_source in self.lidar_sources
            ]
        else:
            data_model[DatasetTableSchema.LIDAR_SOURCES.name] = None

        if self.category_mapping:
            data_model[DatasetTableSchema.CATEGORY_MAPPING.name] = (
                self.category_mapping.to_dictionary()
            )
        else:
            data_model[DatasetTableSchema.CATEGORY_MAPPING.name] = None

        return data_model

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> DatasetRecord:
        """
        Load the dataset record from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the dataset record, which is
            deserialized from a Polars dataframe.

        Returns:
          DatasetRecord: Data model of the dataset record.
        """
        lidar_frames = data_model[DatasetTableSchema.LIDAR_FRAMES.name]
        lidar_frames = [
            LidarFrameDataModel.load_from_dictionary(lidar_frame) for lidar_frame in lidar_frames
        ]

        lidar_sources = data_model[DatasetTableSchema.LIDAR_SOURCES.name]
        if lidar_sources is not None:
            lidar_sources = [
                LidarSourceDataModel.load_from_dictionary(lidar_source)
                for lidar_source in lidar_sources
            ]
        else:
            lidar_sources = None

        category_mapping = data_model[DatasetTableSchema.CATEGORY_MAPPING.name]
        if category_mapping is not None:
            category_mapping = CategoryMappingDataModel.load_from_dictionary(category_mapping)
        else:
            category_mapping = None

        return cls(
            scenario_id=data_model[DatasetTableSchema.SCENARIO_ID.name],
            sample_id=data_model[DatasetTableSchema.SAMPLE_ID.name],
            sample_index=data_model[DatasetTableSchema.SAMPLE_INDEX.name],
            timestamp_seconds=data_model[DatasetTableSchema.TIMESTAMP_SECONDS.name],
            location=data_model[DatasetTableSchema.LOCATION.name],
            vehicle_type=data_model[DatasetTableSchema.VEHICLE_TYPE.name],
            scenario_name=data_model[DatasetTableSchema.SCENARIO_NAME.name],
            lidar_frames=lidar_frames,
            lidar_sources=lidar_sources,
            category_mapping=category_mapping,
        )
