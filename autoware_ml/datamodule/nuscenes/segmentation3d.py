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

    The records store the semantic mask as a full path, like the lidar frames, because the
    nuScenes layout is too shallow to rebuild it from a fixed number of trailing directories.
    """

    def semantic_mask_path(self, lidar_frame: LidarFrameDataModel) -> str:
        """
        Return the stored semantic mask path, which is already usable as it is.

        Args:
          lidar_frame: Lidar frame carrying a semantic mask.

        Returns:
          str: Path of the mask as stored in the record.
        """
        return lidar_frame.lidar_pointcloud_semantic_mask_path
