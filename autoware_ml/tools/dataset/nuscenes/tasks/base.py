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

"""Base task-annotation generator interfaces for NuScenes.

This module defines the shared protocol implemented by NuScenes task adapters
used during info-file generation.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class TaskAnnotationGenerator(ABC):
    """Abstract base class for task-specific annotation generators.

    Each task (detection3d, segmentation3d, calibration_status) extends this
    to add task-specific annotations to the info dictionary.
    """

    @abstractmethod
    def process_sample(
        self,
        info_dict: dict[str, Any],
        nusc: Any,
        sample: Mapping[str, Any],
        cam_name: str | None = None,
    ) -> dict[str, Any]:
        """Process a sample and add task-specific annotations.

        Args:
            info_dict: Base info dictionary with common fields.
            nusc: NuScenes API instance.
            sample: NuScenes sample dictionary.
            cam_name: Optional camera name if processing camera-specific data.

        Returns:
            Updated info dictionary with task-specific annotations.
        """
        raise NotImplementedError("Subclasses must implement process_sample method")
