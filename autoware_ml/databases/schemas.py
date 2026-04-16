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

from dataclasses import dataclass
from typing import NamedTuple

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

    SCENARIO_ID = DatasetTableColumn("scenario_id", pl.String)
    SAMPLE_ID = DatasetTableColumn("sample_id", pl.String)
    SAMPLE_INDEX = DatasetTableColumn("sample_index", pl.Int32)
    LOCATION = DatasetTableColumn("location", pl.String)
    VEHICLE_TYPE = DatasetTableColumn("vehicle_type", pl.String)
    # List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y
    # BBOX_3D = DatasetTableColumn("bbox_3d", pl.List(pl.Array(pl.Float64, shape=(9))))

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
    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    location: str | None
    vehicle_type: str | None
    # TODO (KokSeang): Add more annotation fields here
    # bbox_3d: Sequence[npt.NDArray[np.float64]]
