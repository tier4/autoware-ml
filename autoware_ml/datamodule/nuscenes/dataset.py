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

from autoware_ml.datamodule.t4dataset.dataset import T4Dataset


class NuScenesDataset(T4Dataset):
    """
    Dataset serving every task from the record table of a nuScenes database.

    The nuScenes record generator stores every frame path relative to the database root, so a
    stored path is joined onto the root of this dataset. Unlike the T4 layout, the nuScenes
    layout is too shallow to rebuild a path from a fixed number of trailing directories, so the
    whole stored path is used.
    """

    def _update_frame_path(self, frame_path: str) -> str:
        """
        Resolve a stored frame path against the database root.

        Args:
          frame_path: Frame path relative to the database root, as the record table stores it.

        Returns:
          str: Path of the frame under the database root.
        """

        return str(self.database_root_path / frame_path)
