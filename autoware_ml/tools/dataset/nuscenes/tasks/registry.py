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

"""Registry of NuScenes task-annotation generators."""

from autoware_ml.tools.dataset.nuscenes.tasks.base import TaskAnnotationGenerator
from autoware_ml.tools.dataset.nuscenes.tasks.calibration_status import CalibrationStatusTask
from autoware_ml.tools.dataset.nuscenes.tasks.detection3d import Detection3DTask
from autoware_ml.tools.dataset.nuscenes.tasks.segmentation3d import Segmentation3DTask

TASK_REGISTRY: dict[str, type[TaskAnnotationGenerator]] = {
    "calibration_status": CalibrationStatusTask,
    "detection3d": Detection3DTask,
    "segmentation3d": Segmentation3DTask,
}


def register_task(task_name: str, task_class: type[TaskAnnotationGenerator]) -> None:
    """Register a task annotation generator."""
    TASK_REGISTRY[task_name] = task_class


def get_task(task_name: str) -> type[TaskAnnotationGenerator]:
    """Return the registered task-annotation generator class."""
    if task_name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return TASK_REGISTRY[task_name]


def create_task(task_name: str) -> TaskAnnotationGenerator:
    """Instantiate the requested task-annotation generator."""
    return get_task(task_name)()
