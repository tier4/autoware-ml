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

"""Training entrypoint for Autoware-ML models.

This script wires Hydra configuration, Lightning runtime setup, MLflow
integration, and trainer execution for model training.
"""

import logging

import hydra
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    get_config_path,
    instantiate_callbacks,
    instantiate_trainer,
    log_configuration,
    log_hyperparameters,
    resolve_work_dir,
    set_seed,
)
from autoware_ml.utils.mlflow import (
    build_run_tags,
    build_training_metadata,
    configure_logger,
    generate_experiment_name,
    generate_run_name,
    get_user_config_name,
    log_path_as_artifact,
    should_enable_logger,
    write_run_metadata,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    log_configuration(cfg)
    work_dir = resolve_work_dir()
    logger.info(f"Working directory: {work_dir}")
    config_name = get_user_config_name()
    logger_enabled = should_enable_logger(cfg)

    configure_torch_runtime()
    set_seed(cfg)

    logger.info("Instantiating datamodule...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info("Instantiating model...")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg, logger_enabled=logger_enabled)

    logger.info("Instantiating loggers...")
    trainer_logger = None
    experiment_name = generate_experiment_name(config_name)
    run_name = generate_run_name(config_name, work_dir, "train")
    run_tags = build_run_tags(config_name, work_dir, "train")
    if logger_enabled:
        configure_logger(cfg.logger, experiment_name, run_name, run_tags)
        trainer_logger = hydra.utils.instantiate(cfg.logger)

    logger.info("Instantiating trainer...")
    trainer: L.Trainer = instantiate_trainer(cfg, callbacks, trainer_logger, work_dir)

    log_hyperparameters(cfg, trainer_logger, trainer)

    # Start training
    logger.info("Starting training...")
    logger.info(f"Max epochs: {cfg.trainer.max_epochs}")
    logger.info(f"Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    logger.info(f"Devices: {cfg.trainer.get('devices', 'auto')}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint", None))

    if isinstance(trainer_logger, MLFlowLogger):
        metadata_path = write_run_metadata(
            work_dir,
            build_training_metadata(trainer_logger, experiment_name, config_name, work_dir),
        )
        client = trainer_logger.experiment
        log_path_as_artifact(client, trainer_logger.run_id, work_dir / ".hydra", "hydra")
        log_path_as_artifact(client, trainer_logger.run_id, metadata_path, "metadata")
        log_path_as_artifact(client, trainer_logger.run_id, work_dir / "checkpoints", "checkpoints")

    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {work_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
