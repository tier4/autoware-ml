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

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset


class NuScenesDataset(T4Dataset):
    """
    Dataset serving every task from the record table of a nuScenes database.

    The records store every frame path relative to the database root, as the T4 records do,
    so the frames are read the same way. The nuScenes layout keeps the files of every scene
    together under ``samples/`` and ``sweeps/`` rather than in a directory per scene, so the
    scene of a frame cannot be read off its path and comes from the scene name of the record.
    """

    def scene_token(self, idx: int, lidar_frame: LidarFrameDataModel) -> str:
        """
        Scene token of a record, the scene name the nuScenes records generator keys it by.

        Args:
          idx: Index of the record.
          lidar_frame: Keyframe lidar frame of the record, unused: the nuScenes paths carry
            no scene.

        Returns:
          str: Scene name of the record, such as ``scene-0061``.
        """
        return self.dataset_records_dataframe.item(idx, DatasetTableSchema.SCENARIO_ID.name)
