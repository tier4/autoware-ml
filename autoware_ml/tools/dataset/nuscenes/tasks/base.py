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

"""Base task annotation generator interface for NuScenes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TaskAnnotationGenerator(ABC):
    """Abstract base class for task-specific annotation generators.

    Each task (detection3d, segmentation3d, calibration_status) extends this
    to add task-specific annotations to the info dictionary.
    """

    @abstractmethod
    def process_sample(
        self,
        info_dict: Dict[str, Any],
        nusc: Any,
        sample: Dict[str, Any],
        cam_name: Optional[str] = None,
    ) -> Dict[str, Any]:
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
