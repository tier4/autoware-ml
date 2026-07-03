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

    The unified per-frame info schema already stores every camera's
    calibration (``cam2img``, ``lidar2cam``, ``distortion_coefficients``,
    ``distortion_model``) under ``images``. The calibration datamodule expands
    one record per (frame, camera) at load time, so no per-frame annotation is
    required here; this task is a no-op retained for the task interface.
    """

    def process_sample(
        self,
        info_dict: dict[str, Any],
        nusc: Any,
        sample: Mapping[str, Any],
        cam_name: Any = None,
    ) -> dict[str, Any]:
        """Return the info dict unchanged.

        Args:
            info_dict: Base info dictionary with the unified ``images`` schema.
            nusc: NuScenes API instance (unused; retained for interface compat).
            sample: NuScenes sample dictionary (unused).
            cam_name: Not used.

        Returns:
            The unchanged info dictionary.
        """
        return info_dict
