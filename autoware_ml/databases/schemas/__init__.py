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

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt


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

    def __model_post_init__(self, __context: Any) -> None:
        """Validate that all attributes are of the same length."""
        assert (
            len(self.indices)
            == len(self.boxes_3d)
            == len(self.boxes_3d_instance_ids)
            == len(self.boxes_3d_dataset_names)
            == len(self.boxes_3d_label_names)
            == len(self.boxes_3d_label_indices)
            == len(self.boxes_3d_num_lidar_pointclouds)
            == len(self.boxes_3d_num_radar_pointclouds)
            == len(self.boxes_3d_valid)
            == len(self.boxes_3d_attributes)
        ), "All attributes must be of the same length"
