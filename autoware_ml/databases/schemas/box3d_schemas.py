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
from typing import Set, Any, Mapping

from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt
import polars as pl

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DataModelInterface,
    DatasetTableColumn,
)
from autoware_ml.types.geometry import Box3DFieldIndex


@dataclass(frozen=True)
class Box3DDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to 3D bounding boxes.
    """

    BOX3D_PARAMS = DatasetTableColumn(
        "box3d_params", pl.Array(pl.Float32, shape=(len(Box3DFieldIndex),))
    )
    BOX3D_INSTANCE_ID = DatasetTableColumn("box3d_instance_id", pl.String)
    BOX3D_DATASET_LABEL_NAME = DatasetTableColumn("box3d_dataset_label_name", pl.String)
    BOX3D_LABEL_NAME = DatasetTableColumn("box3d_label_name", pl.String)
    BOX3D_LABEL_INDEX = DatasetTableColumn("box3d_label_index", pl.Int32)
    BOX3D_NUM_LIDAR_POINTCLOUDS = DatasetTableColumn("box3d_num_lidar_pointclouds", pl.Int32)
    BOX3D_NUM_RADAR_POINTCLOUDS = DatasetTableColumn("box3d_num_radar_pointclouds", pl.Int32)
    BOX3D_VALID = DatasetTableColumn("box3d_valid", pl.Boolean)
    BOX3D_ATTRIBUTES = DatasetTableColumn("box3d_attributes", pl.List(pl.String))
    BOX3D_COORDINATE = DatasetTableColumn("box3d_coordinate", pl.String)


class Box3DDataModel(BaseModel, DataModelInterface):
    """
    Data model to represent annotation data for a 3D bounding box.

    Attributes:
      box3d_params: 3D bounding box parameters. It is in (x, y, z, length, width,
        height, yaw, velocity_x, velocity_y, velocity_z) format, following the Box3DFieldIndex enumeration.
      box3d_instance_id: Instance ID for the 3D bounding box.
      box3d_dataset_label_name: Dataset label name for the 3D bounding box.
      box3d_label_name: Label name for the 3D bounding box.
      box3d_label_index: Label index for the 3D bounding box.
      box3d_num_lidar_pointclouds: Number of lidar pointclouds for the 3D bounding box.
      box3d_num_radar_pointclouds: Number of radar pointclouds for the 3D bounding box.
      box3d_valid: Valid flag for the 3D bounding box.
      box3d_attributes: Attributes for the 3D bounding box.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    box3d_params: npt.NDArray[np.float64]
    box3d_instance_id: str
    box3d_dataset_label_name: str
    box3d_label_name: str
    box3d_label_index: int
    box3d_num_lidar_pointclouds: int
    box3d_num_radar_pointclouds: int
    box3d_valid: bool
    box3d_attributes: Set[str]
    box3d_coordinate: str

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the category mapping data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the category mapping data model.
        """
        return {
            Box3DDatasetSchema.BOX3D_PARAMS.name: self.box3d_params_fp32,
            Box3DDatasetSchema.BOX3D_INSTANCE_ID.name: self.box3d_instance_id,
            Box3DDatasetSchema.BOX3D_DATASET_LABEL_NAME.name: self.box3d_dataset_label_name,
            Box3DDatasetSchema.BOX3D_LABEL_NAME.name: self.box3d_label_name,
            Box3DDatasetSchema.BOX3D_LABEL_INDEX.name: self.box3d_label_index,
            Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTCLOUDS.name: self.box3d_num_lidar_pointclouds,
            Box3DDatasetSchema.BOX3D_NUM_RADAR_POINTCLOUDS.name: self.box3d_num_radar_pointclouds,
            Box3DDatasetSchema.BOX3D_VALID.name: self.box3d_valid,
            Box3DDatasetSchema.BOX3D_ATTRIBUTES.name: sorted(list(self.box3d_attributes)),
            Box3DDatasetSchema.BOX3D_COORDINATE.name: self.box3d_coordinate,
        }

    @property
    def box3d_params_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the box3d_params to float32.

        Returns:
          npt.NDArray[np.float32]: Box3D parameters in float32.
        """
        return self.box3d_params.astype(np.float32)

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> Box3DDataModel:
        """
        Load the category mapping data model and decode it to the corresponding CategoryMappingDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the category mapping data model, which is
          deserialized from a Polars dataframe.
        """
        return cls(
            box3d_params=np.asarray(
                data_model[Box3DDatasetSchema.BOX3D_PARAMS.name], dtype=np.float64
            ),
            box3d_instance_id=data_model[Box3DDatasetSchema.BOX3D_INSTANCE_ID.name],
            box3d_dataset_label_name=data_model[Box3DDatasetSchema.BOX3D_DATASET_LABEL_NAME.name],
            box3d_label_name=data_model[Box3DDatasetSchema.BOX3D_LABEL_NAME.name],
            box3d_label_index=data_model[Box3DDatasetSchema.BOX3D_LABEL_INDEX.name],
            box3d_num_lidar_pointclouds=data_model[
                Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTCLOUDS.name
            ],
            box3d_num_radar_pointclouds=data_model[
                Box3DDatasetSchema.BOX3D_NUM_RADAR_POINTCLOUDS.name
            ],
            box3d_valid=data_model[Box3DDatasetSchema.BOX3D_VALID.name],
            box3d_attributes=set(data_model[Box3DDatasetSchema.BOX3D_ATTRIBUTES.name]),
            box3d_coordinate=data_model[Box3DDatasetSchema.BOX3D_COORDINATE.name],
        )

    def create_new_data_model(
        self,
        box3d_params: npt.NDArray[np.float32] | None = None,
        box3d_instance_id: str | None = None,
        box3d_dataset_label_name: str | None = None,
        box3d_label_name: str | None = None,
        box3d_label_index: int | None = None,
        box3d_num_lidar_pointclouds: int | None = None,
        box3d_num_radar_pointclouds: int | None = None,
        box3d_valid: bool | None = None,
        box3d_attributes: Set[str] | None = None,
        box3d_coordinate: str | None = None,
    ) -> Box3DDataModel:
        """
        Create a new Boxes3DDataModel object with the given attributes.
        """
        return Box3DDataModel(
            box3d_params=box3d_params if box3d_params is not None else self.box3d_params,
            box3d_instance_id=box3d_instance_id
            if box3d_instance_id is not None
            else self.box3d_instance_id,
            box3d_dataset_label_name=box3d_dataset_label_name
            if box3d_dataset_label_name is not None
            else self.box3d_dataset_label_name,
            box3d_label_name=box3d_label_name
            if box3d_label_name is not None
            else self.box3d_label_name,
            box3d_label_index=box3d_label_index
            if box3d_label_index is not None
            else self.box3d_label_index,
            box3d_num_lidar_pointclouds=box3d_num_lidar_pointclouds
            if box3d_num_lidar_pointclouds is not None
            else self.box3d_num_lidar_pointclouds,
            box3d_num_radar_pointclouds=box3d_num_radar_pointclouds
            if box3d_num_radar_pointclouds is not None
            else self.box3d_num_radar_pointclouds,
            box3d_valid=box3d_valid if box3d_valid is not None else self.box3d_valid,
            box3d_attributes=box3d_attributes
            if box3d_attributes is not None
            else self.box3d_attributes,
            box3d_coordinate=box3d_coordinate
            if box3d_coordinate is not None
            else self.box3d_coordinate,
        )
