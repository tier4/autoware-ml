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

from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt
import polars as pl

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DataModelInterface,
    DatasetTableColumn,
)
from autoware_ml.common.enums.enums import Box3DFieldIndex


@dataclass(frozen=True)
class Boxes3DDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to category mapping.
    """

    BOXES_3D_ARRAYS = DatasetTableColumn(
        "boxes_3d_arrays", pl.List(pl.Array(pl.Float32, shape=(len(Box3DFieldIndex),)))
    )
    BOXES_3D_INSTANCE_IDS = DatasetTableColumn("boxes_3d_instance_ids", pl.List(pl.String))
    BOXES_3D_DATASET_LABEL_NAMES = DatasetTableColumn(
        "boxes_3d_dataset_label_names", pl.List(pl.String)
    )
    BOXES_3D_LABEL_NAMES = DatasetTableColumn("boxes_3d_label_names", pl.List(pl.String))
    BOXES_3D_LABEL_INDICES = DatasetTableColumn("boxes_3d_label_indices", pl.List(pl.Int32))
    BOXES_3D_NUM_LIDAR_POINTCLOUDS = DatasetTableColumn(
        "boxes_3d_num_lidar_pointclouds", pl.List(pl.Int32)
    )
    BOXES_3D_NUM_RADAR_POINTCLOUDS = DatasetTableColumn(
        "boxes_3d_num_radar_pointclouds", pl.List(pl.Int32)
    )
    BOXES_3D_VALID = DatasetTableColumn("boxes_3d_valid", pl.List(pl.Boolean))
    BOXES_3D_ATTRIBUTES = DatasetTableColumn("boxes_3d_attributes", pl.List(pl.List(pl.String)))


class Boxes3DDataModel(BaseModel, DataModelInterface):
    """
    Base metadata of 3D bounding boxes that can be used for any 3D bounding box dataset.

    Attributes:
      boxes_3d_arrays: Sequence of 3D bounding boxes arrays. It is in (x, y, z, length, width,
        height, yaw, velocity_x, velocity_y, velocity_z) format, following the Box3DFieldIndex enumeration.
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

    def model_post_init(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""

        num_boxes = len(self.boxes_3d_arrays)
        assert (
            len(self.boxes_3d_instance_ids) == num_boxes
            and len(self.boxes_3d_dataset_label_names) == num_boxes
            and len(self.boxes_3d_label_names) == num_boxes
            and len(self.boxes_3d_label_indices) == num_boxes
            and len(self.boxes_3d_num_lidar_pointclouds) == num_boxes
            and len(self.boxes_3d_num_radar_pointclouds) == num_boxes
            and len(self.boxes_3d_valid) == num_boxes
            and len(self.boxes_3d_attributes) == num_boxes
        ), "All attributes must be of the same length"

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the category mapping data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the category mapping data model.
        """

        return self.model_dump()

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> Boxes3DDataModel:
        """
        Load the category mapping data model and decode it to the corresponding CategoryMappingDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the category mapping data model, which is
          deserialized from a Polars dataframe.
        """
        return cls(
            boxes_3d_arrays=data_model[Boxes3DDatasetSchema.BOXES_3D_ARRAYS.name],
            boxes_3d_instance_ids=data_model[Boxes3DDatasetSchema.BOXES_3D_INSTANCE_IDS.name],
            boxes_3d_dataset_label_names=data_model[
                Boxes3DDatasetSchema.BOXES_3D_DATASET_LABEL_NAMES.name
            ],
            boxes_3d_label_names=data_model[Boxes3DDatasetSchema.BOXES_3D_LABEL_NAMES.name],
            boxes_3d_label_indices=data_model[Boxes3DDatasetSchema.BOXES_3D_LABEL_INDICES.name],
            boxes_3d_num_lidar_pointclouds=data_model[
                Boxes3DDatasetSchema.BOXES_3D_NUM_LIDAR_POINTCLOUDS.name
            ],
            boxes_3d_num_radar_pointclouds=data_model[
                Boxes3DDatasetSchema.BOXES_3D_NUM_RADAR_POINTCLOUDS.name
            ],
            boxes_3d_valid=data_model[Boxes3DDatasetSchema.BOXES_3D_VALID.name],
            boxes_3d_attributes=data_model[Boxes3DDatasetSchema.BOXES_3D_ATTRIBUTES.name],
        )

    def remove_boxes(self, indices: Sequence[int]) -> Boxes3DDataModel:
        """
        Remove the boxes at the given indices, and create a new Boxes3DMetadata object.
        """
        return Boxes3DDataModel(
            boxes_3d_arrays=self.boxes_3d_arrays[indices],
            boxes_3d_instance_ids=self.boxes_3d_instance_ids[indices],
            boxes_3d_dataset_label_names=self.boxes_3d_dataset_label_names[indices],
            boxes_3d_label_names=self.boxes_3d_label_names[indices],
            boxes_3d_label_indices=self.boxes_3d_label_indices[indices],
            boxes_3d_num_lidar_pointclouds=self.boxes_3d_num_lidar_pointclouds[indices],
            boxes_3d_num_radar_pointclouds=self.boxes_3d_num_radar_pointclouds[indices],
            boxes_3d_valid=self.boxes_3d_valid[indices],
            boxes_3d_attributes=self.boxes_3d_attributes[indices],
        )

    def merge_boxes(self, boxes3d_metadata: Boxes3DDataModel) -> Boxes3DDataModel:
        """
        Merge the boxes with the given boxes3d_metadata by extending the list, and create a new Boxes3DMetadata object.
        """
        return Boxes3DDataModel(
            boxes_3d_arrays=self.boxes_3d_arrays.extend(boxes3d_metadata.boxes_3d_arrays),
            boxes_3d_instance_ids=self.boxes_3d_instance_ids.extend(
                boxes3d_metadata.boxes_3d_instance_ids
            ),
            boxes_3d_dataset_label_names=self.boxes_3d_dataset_label_names.extend(
                boxes3d_metadata.boxes_3d_dataset_label_names
            ),
            boxes_3d_label_names=self.boxes_3d_label_names.extend(
                boxes3d_metadata.boxes_3d_label_names
            ),
            boxes_3d_label_indices=self.boxes_3d_label_indices.extend(
                boxes3d_metadata.boxes_3d_label_indices
            ),
            boxes_3d_num_lidar_pointclouds=self.boxes_3d_num_lidar_pointclouds.extend(
                boxes3d_metadata.boxes_3d_num_lidar_pointclouds
            ),
            boxes_3d_num_radar_pointclouds=self.boxes_3d_num_radar_pointclouds.extend(
                boxes3d_metadata.boxes_3d_num_radar_pointclouds
            ),
            boxes_3d_valid=self.boxes_3d_valid.extend(boxes3d_metadata.boxes_3d_valid),
            boxes_3d_attributes=self.boxes_3d_attributes.extend(
                boxes3d_metadata.boxes_3d_attributes
            ),
        )

    def get_ground_plane_speeds(self) -> npt.NDArray[np.float32]:
        """
        Get the speeds in x-y plane (ground plane) of the 3D bounding boxes.

        Returns:
          npt.NDArray[np.float32] (N, ): The speeds in x-y plane (ground plane) of the 3D bounding boxes.
        """
        velocity_xy = np.asarray(
            [
                self.boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_X],
                self.boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_Y],
            ]
        )
        return np.linalg.norm(velocity_xy, axis=0)

    @classmethod
    def create_new_datamodel(
        cls,
        boxes_3d_arrays: npt.NDArray[np.float32],
        boxes_3d_instance_ids: Sequence[str],
        boxes_3d_dataset_label_names: Sequence[str],
        boxes_3d_label_names: Sequence[str],
        boxes_3d_label_indices: Sequence[int],
        boxes_3d_num_lidar_pointclouds: Sequence[int],
        boxes_3d_num_radar_pointclouds: Sequence[int],
        boxes_3d_valid: Sequence[bool],
        boxes_3d_attributes: Sequence[set[str]],
    ) -> Boxes3DDataModel:
        """
        Create a new Boxes3DDataModel object with the given attributes.
        """
        return Boxes3DDataModel(
            boxes_3d_arrays=boxes_3d_arrays,
            boxes_3d_instance_ids=boxes_3d_instance_ids,
            boxes_3d_dataset_label_names=boxes_3d_dataset_label_names,
            boxes_3d_label_names=boxes_3d_label_names,
            boxes_3d_label_indices=boxes_3d_label_indices,
            boxes_3d_num_lidar_pointclouds=boxes_3d_num_lidar_pointclouds,
            boxes_3d_num_radar_pointclouds=boxes_3d_num_radar_pointclouds,
            boxes_3d_valid=boxes_3d_valid,
            boxes_3d_attributes=boxes_3d_attributes,
        )
