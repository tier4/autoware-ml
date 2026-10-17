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

import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from autoware_ml.datamodule.data_module import DataModule

logger = logging.getLogger(__name__)


def build_datamodule(cfg: DictConfig) -> DataModule:
    """
    Build a DataModule from the Hydra configuration.

    Every dataset source of the datamodule names its own database in the config, so the
    databases are instantiated as part of the datamodule and read back through its
    ``databases()``.

    Args:
        cfg: Hydra configuration
    Returns:
        A DataModule object.
    """
    logger.info("Building datamodule...")
    datamodule = instantiate(cfg.datamodule)
    logger.info(
        "Datamodule built successfully over "
        f"{[database.version for database in datamodule.databases()]}."
    )
    return datamodule
