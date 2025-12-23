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

"""Task registry for NuScenes dataset generation."""

from typing import Dict, Type

from autoware_ml.tools.dataset.nuscenes.tasks.base import TaskAnnotationGenerator
from autoware_ml.tools.dataset.nuscenes.tasks.calibration_status import (
    CalibrationStatusTask,
)
from autoware_ml.tools.dataset.nuscenes.tasks.detection3d import Detection3DTask
from autoware_ml.tools.dataset.nuscenes.tasks.segmentation3d import Segmentation3DTask

_TASK_REGISTRY: Dict[str, Type[TaskAnnotationGenerator]] = {}


def register_task(task_name: str, task_class: Type[TaskAnnotationGenerator]) -> None:
    """Register a task annotation generator.

    Args:
        task_name: Name of the task (e.g., 'detection3d', 'calibration_status').
        task_class: Task class that extends TaskAnnotationGenerator.
    """
    _TASK_REGISTRY[task_name] = task_class


def get_task(task_name: str) -> Type[TaskAnnotationGenerator]:
    """Get a task annotation generator class by name.

    Args:
        task_name: Name of the task.

    Returns:
        Task class.

    Raises:
        ValueError: If task is not registered.
    """
    if task_name not in _TASK_REGISTRY:
        available = ", ".join(_TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return _TASK_REGISTRY[task_name]


def create_task(task_name: str) -> TaskAnnotationGenerator:
    """Create an instance of a task annotation generator.

    Args:
        task_name: Name of the task.

    Returns:
        Task instance.
    """
    task_class = get_task(task_name)
    return task_class()


register_task("detection3d", Detection3DTask)
register_task("segmentation3d", Segmentation3DTask)
register_task("calibration_status", CalibrationStatusTask)
