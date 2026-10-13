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

"""Generate the record table of a database.

The configured database reads the annotations of its scenarios and writes them to the
record table named after its hash. Training generates a missing table itself, this
entrypoint builds it ahead of time.
"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.utils.runtime import get_config_path

logger = logging.getLogger(__name__)
register_config_resolvers()
_CONFIG_PATH = get_config_path()


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """
    Generate the record table of the configured database.

    Args:
        cfg: Hydra configuration
    """

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    database: DatabaseInterface = instantiate(cfg.database)
    database.process_scenario_records()
    logger.info(f"Record table of {database.version}: {database.cache_file_path}")


if __name__ == "__main__":
    main()
