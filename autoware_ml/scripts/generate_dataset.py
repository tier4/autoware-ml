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

from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import autoware_ml.configs
from autoware_ml.databases.database_interface import DatabaseInterface

logger = logging.getLogger(__name__)

_CONFIG_PATH = str(Path(autoware_ml.configs.__file__).parent.resolve() / "generators")


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """
    Script to generate records, and it will be removed in the future.

    Args:
        cfg: Hydra configuration
    """

    # Print configuration
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Instantiate DatabaseInterface
    database: DatabaseInterface = instantiate(cfg.database)

    # Process scenario records and save them to a parquet file
    database.process_scenario_records()


if __name__ == "__main__":
    main()
