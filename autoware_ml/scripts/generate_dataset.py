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
    """Script to generate records, and it will be removed in the future.

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
