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
import os

import hydra
import lightning as L
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
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_RUN_ID_ENV,
    build_run_metadata,
    configure_logger,
    get_user_config_name,
    load_run_context,
    prepare_run_context,
    should_enable_logger,
    write_run_config_artifacts,
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
    run_context = None
    if logger_enabled:
        pre_created_run_id = os.environ.get(AUTOWARE_ML_RUN_ID_ENV)
        if pre_created_run_id is not None:
            run_context = load_run_context(cfg.logger.tracking_uri, pre_created_run_id)
            if work_dir != run_context.hydra_dir:
                raise RuntimeError(
                    f"Hydra work directory '{work_dir}' does not match the pre-created MLflow "
                    f"run directory '{run_context.hydra_dir}'."
                )
        else:
            run_context = prepare_run_context(
                cfg.logger.tracking_uri,
                config_name,
                hydra_dir=work_dir,
                stage="train",
            )

    configure_torch_runtime()
    set_seed(cfg)

    logger.info("Instantiating datamodule...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info("Instantiating model...")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(
        cfg,
        logger_enabled=logger_enabled,
        checkpoint_dir=run_context.checkpoints_dir if run_context is not None else None,
    )

    logger.info("Instantiating loggers...")
    trainer_logger = None
    if logger_enabled:
        write_run_config_artifacts(cfg, run_context.artifact_dir)
        write_run_metadata(
            run_context.artifact_dir,
            build_run_metadata(run_context, config_name, run_context.hydra_dir, "train"),
        )
        configure_logger(
            cfg.logger,
            run_context.experiment_name,
            run_context.run_name,
            run_context.tags,
            run_id=run_context.run_id,
        )
        trainer_logger = hydra.utils.instantiate(cfg.logger)

    logger.info("Instantiating trainer...")
    trainer: L.Trainer = instantiate_trainer(
        cfg,
        callbacks,
        trainer_logger,
        run_context.artifact_dir if run_context is not None else work_dir,
    )

    log_hyperparameters(cfg, trainer_logger, trainer)

    # Start training
    logger.info("Starting training...")
    logger.info(f"Max epochs: {cfg.trainer.max_epochs}")
    logger.info(f"Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    logger.info(f"Devices: {cfg.trainer.get('devices', 'auto')}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint", None))

    logger.info("Training completed!")
    checkpoints_dir = (
        run_context.checkpoints_dir if run_context is not None else work_dir / "checkpoints"
    )
    logger.info(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == "__main__":
    main()
