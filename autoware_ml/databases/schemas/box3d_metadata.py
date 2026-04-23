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

from typing import Sequence

from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt

from autoware_ml.common.enums.enums import Box3DFieldIndex


class Boxes3DMetadata(BaseModel):
    """
    Base metadata of 3D bounding boxes that can be used for any 3D bounding box dataset.

    Attributes:
      boxes_3d_arrays: Sequence of 3D bounding boxes arrays.
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

    def remove_boxes(self, indices: Sequence[int]) -> Boxes3DMetadata:
        """
        Remove the boxes at the given indices, and create a new Boxes3DMetadata object.
        """
        return Boxes3DMetadata(
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

    def merge_boxes(self, boxes3d_metadata: Boxes3DMetadata) -> Boxes3DMetadata:
        """
        Merge the boxes with the given boxes3d_metadata by extending the list, and create a new Boxes3DMetadata object.
        """
        return Boxes3DMetadata(
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

    def get_bev_speeds(self) -> npt.NDArray[np.float32]:
        """
        Get the speeds in x-y plane (BEV) of the 3D bounding boxes.

        Returns:
          npt.NDArray[np.float32] (N, ): The speeds in x-y plane (BEV) of the 3D bounding boxes.
        """
        velocity_xy = np.asarray(
            [
                self.boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_X],
                self.boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_Y],
            ]
        )
        return np.linalg.norm(velocity_xy, axis=0)

    @classmethod
    def create_new_metadata(
        cls,
        boxes_3d_arrays: npt.NDArray[np.float32] | None,
        boxes_3d_instance_ids: Sequence[str] | None,
        boxes_3d_dataset_label_names: Sequence[str] | None,
        boxes_3d_label_names: Sequence[str] | None,
        boxes_3d_label_indices: Sequence[int] | None,
        boxes_3d_num_lidar_pointclouds: Sequence[int] | None,
        boxes_3d_num_radar_pointclouds: Sequence[int] | None,
        boxes_3d_valid: Sequence[bool] | None,
        boxes_3d_attributes: Sequence[set[str]] | None,
    ) -> Boxes3DMetadata:
        """
        Create a new Boxes3DMetadata object with the given attributes.
        """
        return Boxes3DMetadata(
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
