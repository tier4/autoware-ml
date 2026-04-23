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
from typing import Sequence, Any, Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DatasetTableColumn,
    DataModelInterface,
)


@dataclass(frozen=True)
class CategoryMappingDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to category mapping.
    """

    CATEGORY_NAMES = DatasetTableColumn("category_names", pl.List(pl.String))
    CATEGORY_INDICES = DatasetTableColumn("category_indices", pl.List(pl.Int32))


class CategoryMappingDataModel(BaseModel, DataModelInterface):
    """
    Category mapping data model that can be shared by multiple datasets.

    Attributes:
      category_name: Category name.
      category_index: Category index.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    category_names: Sequence[str]
    category_indices: Sequence[int]

    def model_post_init(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""

        assert len(self.category_names) == len(self.category_indices), (
            "All attributes must be of the same length"
        )

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the category mapping data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the category mapping data model.
        """

        return self.model_dump()

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> CategoryMappingDataModel:
        """
        Load the category mapping data model and decode it to the corresponding CategoryMappingDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the category mapping data model, which is
          deserialized from a Polars dataframe.
        """
        return cls(
            category_names=data_model[CategoryMappingDatasetSchema.CATEGORY_NAMES.name],
            category_indices=data_model[CategoryMappingDatasetSchema.CATEGORY_INDICES.name],
        )
