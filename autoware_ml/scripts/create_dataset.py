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

"""Dataset generation script."""

import logging
from typing import Any, Dict, List

from autoware_ml.tools.dataset.nuscenes import NuScenesDatasetGenerator

logger = logging.getLogger(__name__)

_DATASET_GENERATORS: Dict[str, Any] = {
    "nuscenes": NuScenesDatasetGenerator,
}


def main(
    dataset: str,
    tasks: List[str],
    root_path: str,
    out_dir: str,
    **kwargs: Any,
) -> None:
    """Generate dataset info files.

    Args:
        dataset: Dataset name (e.g., 'nuscenes').
        tasks: List of task names to generate annotations for.
        root_path: Root path of the dataset.
        out_dir: Output directory for info files.
        **kwargs: Dataset-specific arguments (e.g., version, max_sweeps).
    """
    if dataset not in _DATASET_GENERATORS:
        available = ", ".join(_DATASET_GENERATORS.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Available datasets: {available}")

    generator_class = _DATASET_GENERATORS[dataset]
    generator = generator_class()

    logger.info(f"Generating {dataset} dataset with tasks: {tasks}")
    logger.info(f"Root path: {root_path}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Dataset-specific args: {kwargs}")

    generator.generate(root_path=root_path, out_dir=out_dir, tasks=tasks, **kwargs)

    logger.info("Dataset generation completed successfully")
