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

from autoware_ml.datamodule.multi_task.t4dataset.multi_task_t4dataset import MultiTaskT4Dataset


class MultiTaskNuScenesDataset(MultiTaskT4Dataset):
    """
    A dataset class that supports multiple tasks for NuScenesDataset-generated parquet records.

    Only overrides `_update_lidar_pointcloud_path`: MultiTaskT4Dataset's version re-resolves lidar
    paths as `database_root_path / "/".join(path.split("/")[-6:])`, which assumes the stored path
    is deep enough that the last 6 segments never dip into `database_root_path` itself. That holds
    for T4Dataset's own directory layout (several nested levels per scenario), but not for
    NuScenes' much shallower `<dataroot>/samples/<channel>/<file>` layout, where taking the last 6
    segments re-includes `database_root_path` and duplicates it when reconstructing the path.

    NuScenesRecordsGenerator already stores lidar pointcloud paths as full, directly-usable
    absolute paths (via nuscenes-devkit's `NuScenes.get_sample_data`/`get_sample_data_path`), so no
    path reconstruction is needed here at all.
    """

    def _update_lidar_pointcloud_path(self, lidar_pointcloud_path: str) -> str:
        """
        Return the lidar pointcloud path unchanged, since NuScenesRecordsGenerator already stores
        full, directly-usable absolute paths.

        Args:
          lidar_pointcloud_path: Lidar pointcloud path as stored in the parquet record.

        Returns:
          str: The same path, unmodified.
        """

        return lidar_pointcloud_path
