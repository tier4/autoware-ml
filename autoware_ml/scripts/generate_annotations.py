import logging

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

import autoware_ml.configs

logger = logging.getLogger(__name__)

_CONFIG_PATH = str(Path(autoware_ml.configs.__file__).parent.resolve() / "annotations")


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


if __name__ == "__main__":
    main()
