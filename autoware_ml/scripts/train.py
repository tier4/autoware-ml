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
from pathlib import Path

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
    build_dataset_metadata,
    build_dataset_tags,
    build_run_tags,
    build_training_metadata,
    configure_logger,
    create_child_run,
    generate_run_name,
    get_user_config_name,
    log_coco_dataset_inputs,
    log_path_as_artifact,
    reopen_run,
    resolve_lineage_context,
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
    checkpoint_path = cfg.get("checkpoint", None)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    mlflow_resume_run = bool(cfg.get("mlflow_resume_run", False))
    dataset_metadata = build_dataset_metadata(cfg)
    experiment_name, parent_run_id = resolve_lineage_context(
        config_name,
        checkpoint_path,
        tracking_uri=cfg.logger.tracking_uri if logger_enabled else None,
    )
    if mlflow_resume_run and checkpoint_path is not None and parent_run_id is None:
        raise ValueError(
            "Same-run MLflow resume requires checkpoint metadata with an originating run_id."
        )
    run_name = generate_run_name(config_name, work_dir, "train")
    run_tags = build_run_tags(
        config_name,
        work_dir,
        "train",
        extra_tags={
            **build_dataset_tags(cfg),
            **({"checkpoint_path": str(checkpoint_path)} if checkpoint_path is not None else {}),
            **({"source_run_id": parent_run_id} if parent_run_id and not mlflow_resume_run else {}),
            **({"resume_mode": "same_run"} if mlflow_resume_run else {}),
        },
    )
    if logger_enabled:
        if mlflow_resume_run and parent_run_id is not None:
            run_id = parent_run_id
            reopen_run(cfg.logger.tracking_uri, run_id, tags=run_tags)
        else:
            run_id = create_child_run(
                cfg.logger.tracking_uri,
                experiment_name,
                run_name,
                run_tags,
                parent_run_id,
            )
        configure_logger(cfg.logger, experiment_name, run_name, run_tags, run_id=run_id)
        trainer_logger = hydra.utils.instantiate(cfg.logger)
        if isinstance(trainer_logger, MLFlowLogger) and not mlflow_resume_run:
            log_coco_dataset_inputs(trainer_logger.experiment, trainer_logger.run_id, cfg, "train", work_dir)

    logger.info("Instantiating trainer...")
    trainer: L.Trainer = instantiate_trainer(cfg, callbacks, trainer_logger, work_dir)

    if mlflow_resume_run and isinstance(trainer_logger, MLFlowLogger):
        logger.info("Skipping hyperparameter re-logging for same-run MLflow resume.")
    else:
        log_hyperparameters(cfg, trainer_logger, trainer)

    # Start training
    logger.info("Starting training...")
    logger.info(f"Max epochs: {cfg.trainer.max_epochs}")
    logger.info(f"Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    logger.info(f"Devices: {cfg.trainer.get('devices', 'auto')}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=str(checkpoint_path) if checkpoint_path else None)

    if isinstance(trainer_logger, MLFlowLogger):
        metadata_path = write_run_metadata(
            work_dir,
            build_training_metadata(
                trainer_logger,
                experiment_name,
                config_name,
                work_dir,
                source_run_id=(
                    parent_run_id
                    if parent_run_id is not None and parent_run_id != trainer_logger.run_id
                    else None
                ),
                checkpoint_path=checkpoint_path,
                dataset_metadata=dataset_metadata,
            ),
        )
        client = trainer_logger.experiment
        log_path_as_artifact(client, trainer_logger.run_id, work_dir / ".hydra", "hydra")
        log_path_as_artifact(client, trainer_logger.run_id, metadata_path, "metadata")
        log_path_as_artifact(client, trainer_logger.run_id, work_dir / "checkpoints", "checkpoints")

    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {work_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
