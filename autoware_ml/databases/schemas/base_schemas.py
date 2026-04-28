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

from abc import ABC, abstractmethod
from typing import NamedTuple, Sequence, Mapping, Any

import polars as pl


class DatasetTableColumn(NamedTuple):
    """
    Annotation table column.

    Attributes:
      name: Name of the column.
      dtype: Data type of the column.
    """

    name: str
    dtype: pl.DataType


class BaseFieldSchema:
    """
    Base class for field schemas.
    """

    @classmethod
    def to_polars_field_schema(cls) -> Sequence[pl.Field]:
        """
        Convert the lidar column schema to a Polars field schema.

        Returns:
          pl.Schema: Polars schema.
        """

        return [
            pl.Field(v.name, v.dtype)
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and isinstance(v, DatasetTableColumn)
        ]


class DataModelInterface(ABC):
    """
    Interface for data models.
    """

    @abstractmethod
    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the data model.
        """

        raise NotImplementedError("Subclasses must implement to_dictionary!")

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> DataModelInterface:
        """
        Load the data model and decode it to the corresponding data model from a dictionary, which is
        deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the data model, which is
          deserialized from a Polars dataframe.

        Returns:
          DataModelInterface: Data model.
        """

        raise NotImplementedError("Subclasses must implement load_from_dictionary!")
