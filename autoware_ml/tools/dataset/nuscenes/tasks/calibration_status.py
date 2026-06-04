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

"""Calibration-status annotation generator for NuScenes info creation.

This module injects calibration-status labels and related metadata into
generated NuScenes sample dictionaries.
"""

from collections.abc import Mapping
from typing import Any

from autoware_ml.tools.dataset.nuscenes.tasks.base import TaskAnnotationGenerator


class CalibrationStatusTask(TaskAnnotationGenerator):
    """Task generator for calibration status annotations.

    Creates separate sample entries for each camera by reading pre-computed
    calibration fields from the info dict's unified ``"images"`` schema.
    """

    def process_sample(
        self,
        info_dict: dict[str, Any],
        nusc: Any,
        sample: Mapping[str, Any],
        cam_name: Any = None,
    ) -> dict[str, Any]:
        """Add calibration_status annotations to the info dict.

        For calibration_status, we create separate samples for each camera.
        This method adds a special marker that the generator will use to expand
        into multiple samples.

        Args:
            info_dict: Base info dictionary. Must contain an ``"images"`` key
                populated by the NuScenes generator with the unified schema.
            nusc: NuScenes API instance (unused; retained for interface compat).
            sample: NuScenes sample dictionary.
            cam_name: Not used.

        Returns:
            Updated info dictionary with calibration_status annotations.
        """
        calibration_samples = []
        for cam, cam_info in info_dict.get("images", {}).items():
            calibration_sample = {
                "image": {
                    "img_path": cam_info["img_path"],
                    "cam2img": cam_info["cam2img"],
                    "lidar2cam": cam_info["lidar2cam"],
                    "distortion_model": cam_info.get("distortion_model", ""),
                    "distortion_coeffs": cam_info.get("distortion_coeffs", []),
                },
                "lidar_points": {
                    "lidar_path": info_dict["lidar_path"],
                },
                "calibration_status_task": True,
                "camera_name": cam,
            }
            calibration_samples.append(calibration_sample)

        info_dict["calibration_status_samples"] = calibration_samples
        return info_dict
