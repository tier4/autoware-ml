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

from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DTask


class NuScenesSegmentation3DTask(T4Segmentation3DTask):
    """
    3D segmentation task for NuScenesDataset records.

    The records store the semantic mask path relative to the database root, like the lidar
    frames, and the whole stored path is joined onto the root: the nuScenes layout is too
    shallow to rebuild it from a fixed number of trailing directories as the T4 task does.
    """

    def semantic_mask_path(self, lidar_frame: LidarFrameDataModel) -> str:
        """
        Resolve the stored semantic mask path against the database root.

        Args:
          lidar_frame: Lidar frame carrying a semantic mask.

        Returns:
          str: Path of the mask under the database root.
        """
        return str(self.database_root_path / lidar_frame.lidar_pointcloud_semantic_mask_path)
