import logging

from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import autoware_ml.configs
from autoware_ml.databases.database_interface import DatabaseInterface

logger = logging.getLogger(__name__)

_CONFIG_PATH = str(Path(autoware_ml.configs.__file__).parent.resolve() / "generators")


def build_database(cfg: DictConfig) -> DatabaseInterface:
    """Build database interface.

    Args:
        cfg: Hydra configuration
    """
    # First, set the configuration to be mutable
    OmegaConf.set_struct(cfg, False)
    # Then, extract the scenario configs from the database config
    # Must instantiate separately to reuse the same parameters, for example, scenario_root_path
    for scenario_group, scenario_configs in cfg.database.scenarios.items():
        scenario_configs.scenario_root_path = cfg.database.scenario_root_path
    # cfg.database.scenarios.scenario_root_path = cfg.database.scenario_root_path
    # scenario_configs = cfg.database.pop("scenarios", None)
    # if scenario_configs is None:
    #     raise ValueError("Scenario configs must be provided in the database config.")

    # Set the configuration to be immutable again
    OmegaConf.set_struct(cfg, True)
    logger.info("After:")
    logger.info(OmegaConf.to_yaml(cfg.database))
    # Instantiate the database
    # return instantiate(cfg.database, scenario_configs=scenario_configs)
    return instantiate(cfg.database)


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main training function.

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
    database: DatabaseInterface = build_database(cfg)

    # Process scenario records and save them to a parquet file
    database.process_scenario_records()


if __name__ == "__main__":
    main()
