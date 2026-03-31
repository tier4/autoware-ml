# Copyright 2025 TIER IV, Inc.
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

"""Segmentation3D task annotation generator for NuScenes.

This module defines segmentation-specific NuScenes dataset generation steps used
by the dataset creation tooling.
"""

from collections.abc import Mapping
from typing import Any

from autoware_ml.tools.dataset.nuscenes.tasks.base import TaskAnnotationGenerator


class Segmentation3DTask(TaskAnnotationGenerator):
    """Generate 3D semantic-segmentation annotations for NuScenes info files.

    The task injects lidarseg label paths and related segmentation metadata into
    generated sample dictionaries.
    """

    def process_sample(
        self,
        info_dict: dict[str, Any],
        nusc: Any,
        sample: Mapping[str, Any],
        cam_name: Any = None,
    ) -> dict[str, Any]:
        """Add segmentation3d annotations to the info dict.

        Args:
            info_dict: Base info dictionary.
            nusc: NuScenes API instance.
            sample: NuScenes sample dictionary.
            cam_name: Not used for segmentation3d.

        Returns:
            Updated info dictionary with segmentation3d annotations.
        """
        if "lidarseg" in nusc.table_names:
            lidar_token = sample["data"]["LIDAR_TOP"]
            info_dict["pts_semantic_mask_path"] = nusc.get("lidarseg", lidar_token)["filename"]

        return info_dict
