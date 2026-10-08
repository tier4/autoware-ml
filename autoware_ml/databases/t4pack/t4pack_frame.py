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
from typing import Any, Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DatasetTableColumn,
    DataModelInterface,
)


@dataclass(frozen=True)
class T4PackFrameDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for the location of a lidar frame in its t4pack file.
    """

    OFFSET = DatasetTableColumn("offset", pl.Int64)
    SIZE = DatasetTableColumn("size", pl.Int64)
    NUM_POINTS = DatasetTableColumn("num_points", pl.Int64)
    DTYPES = DatasetTableColumn("dtypes", pl.List(pl.String))


class T4PackFrame(BaseModel, DataModelInterface):
    """
    Location of one lidar frame inside the t4pack file of its channel.

    Attributes:
      offset: Byte offset of the compressed frame in the pack.
      size: Byte size of the compressed frame.
      num_points: Number of points of the frame.
      dtypes: t4pack type code of every column: ``f4s`` byte-shuffled float32, ``u1`` uint8,
        ``i1`` int8.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    offset: int
    size: int
    num_points: int
    dtypes: tuple[str, ...]

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the t4pack frame to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the t4pack frame.
        """

        return {
            T4PackFrameDatasetSchema.OFFSET.name: self.offset,
            T4PackFrameDatasetSchema.SIZE.name: self.size,
            T4PackFrameDatasetSchema.NUM_POINTS.name: self.num_points,
            T4PackFrameDatasetSchema.DTYPES.name: list(self.dtypes),
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> T4PackFrame:
        """
        Load the t4pack frame from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the t4pack frame, which is deserialized
            from a Polars dataframe.

        Returns:
          T4PackFrame: T4PackFrame object.
        """

        return cls(
            offset=int(data_model[T4PackFrameDatasetSchema.OFFSET.name]),
            size=int(data_model[T4PackFrameDatasetSchema.SIZE.name]),
            num_points=int(data_model[T4PackFrameDatasetSchema.NUM_POINTS.name]),
            dtypes=tuple(data_model[T4PackFrameDatasetSchema.DTYPES.name]),
        )
