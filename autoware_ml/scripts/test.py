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

"""Evaluation entrypoint for Autoware-ML models.

This script wires Hydra configuration, Lightning runtime setup, MLflow
integration, and trainer execution for evaluating trained checkpoints.
"""

import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence


import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
from omegaconf import DictConfig
import torch

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context
from autoware_ml.builders.model_builder import (
    build_model,
    build_data_preprocessor,
    build_weight_checkpoint_paths,
)
from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.mlflow_helpers import resolve_lineage_context
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    restrict_console_logging_to_rank_zero,
    get_config_path,
    instantiate_callbacks,
    instantiate_trainer,
    log_configuration,
    log_hyperparameters,
    set_seed,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()
CONFIG_NAME_PREFIX = "tasks/"


def test(
    trainer: L.Trainer,
    cfg: DictConfig,
    model: ModuleBaseModel,
    datamodule: DataModule,
    weight_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """
    Start an evaluation loop.

    Args:
        trainer: Lightning trainer
        cfg: Hydra configuration
        model: Multi-task base model
        datamodule: Multi-task data module
        weight_paths: Checkpoint paths already loaded into the model

    Returns:
        Per-dataloader metric dictionaries reported by the trainer.
    """

    logger.info("Starting evaluation...")
    logger.info(f"Weights: {[str(path) for path in weight_paths]}")
    logger.info(f"Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    logger.info(f"Devices: {cfg.trainer.get('devices', 'auto')}")

    # Weights are applied by the model builder, so the trainer must not reload a checkpoint.
    metrics = trainer.test(model, datamodule=datamodule, ckpt_path=None)
    logger.info("Evaluation completed!")

    return metrics


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    log_configuration(cfg)
    config_name = HydraConfig.get().job.config_name
    if config_name is None:
        raise ValueError("Hydra config name is not available.")

    logger_enabled = cfg.get("logger") is not None
    config_name = config_name.removeprefix(CONFIG_NAME_PREFIX)
    experiment_name = f"{cfg.experiment_group_name}/{cfg.experiment_name}"

    # Configure weights and checkpoint paths
    weight_paths, checkpoint_path = build_weight_checkpoint_paths(cfg)

    # Nest the evaluation under the MLflow run that produced the primary checkpoint, when known
    experiment_name, parent_run_id = resolve_lineage_context(experiment_name, checkpoint_path)

    run_context = build_mlflow_run_context(
        cfg,
        stage="test",
        experiment_name=experiment_name,
        config_name=config_name,
        experiment_uid=cfg.experiment_uid,
        logger_enabled=logger_enabled,
        parent_run_id=parent_run_id,
        extra_tags=MappingProxyType(
            {
                "checkpoint_path": str(checkpoint_path),
                "source_run_id": parent_run_id or "",
            }
        ),
    )

    configure_torch_runtime()
    restrict_console_logging_to_rank_zero()
    set_seed(cfg)

    # Build trainer logger
    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=run_context,
        stage="test",
        config_name=config_name,
        logger_enabled=logger_enabled,
        extra_metadata=MappingProxyType(
            {
                "source_run_id": parent_run_id,
                "checkpoint_path": str(checkpoint_path),
            }
        ),
    )

    # Build datamodule
    datamodule = build_datamodule(cfg)

    # Build model
    data_preprocessor = build_data_preprocessor(cfg)
    model = build_model(
        cfg,
        data_preprocessor=data_preprocessor,
        weights_path=weight_paths,
        resume_checkpoint_path=None,
        device=torch.device("cpu"),
        set_eval=True,
        enforce_full_coverage=True,
    )

    logger.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg, logger_enabled=logger_enabled)

    logger.info("Instantiating trainer...")
    trainer_root_dir = (
        run_context.artifact_dir if run_context is not None else cfg.experiment_run_dir
    )
    trainer: L.Trainer = instantiate_trainer(
        cfg,
        callbacks,
        trainer_logger,
        trainer_root_dir,
    )
    log_hyperparameters(cfg, trainer_logger)

    # Start evaluation
    return test(
        trainer=trainer,
        cfg=cfg,
        model=model,
        datamodule=datamodule,
        weight_paths=weight_paths,
    )


if __name__ == "__main__":
    main()
