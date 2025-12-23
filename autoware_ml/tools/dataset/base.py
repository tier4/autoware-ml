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

"""Base dataset generator interface."""

from abc import ABC, abstractmethod
from typing import Any, List


class DatasetGenerator(ABC):
    """Abstract base class for dataset generators.

    Each dataset (nuscenes, t4dataset, etc.) implements this interface
    to provide a consistent way to generate dataset info files.
    """

    @abstractmethod
    def generate(
        self,
        root_path: str,
        out_dir: str,
        tasks: List[str],
        **kwargs: Any,
    ) -> None:
        """Generate dataset info files.

        Args:
            root_path: Root path of the dataset.
            out_dir: Output directory for info files.
            tasks: List of task names to generate annotations for.
            **kwargs: Dataset-specific arguments (e.g., max_sweeps for nuscenes).
        """
        raise NotImplementedError("Subclasses must implement generate method")
