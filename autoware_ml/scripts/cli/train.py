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

from collections.abc import Sequence
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from omegaconf import DictConfig
import torch

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context
from autoware_ml.builders.model_builder import (
    build_model,
    build_data_preprocessor,
    resolve_weights_paths,
)
from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.mlflow_helpers import MlflowRunContext
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    restrict_console_logging_to_rank_zero,
    get_config_path,
    instantiate_callbacks,
    instantiate_trainer,
    log_configuration,
    log_hyperparameters,
    resolve_work_dir,
    set_seed,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()
CONFIG_NAME_PREFIX = "tasks/"


def fit_model(
    cfg: DictConfig,
    model: ModuleBaseModel,
    datamodule: DataModule,
    run_context: MlflowRunContext | None,
    config_name: str,
    *,
    stage: str = "train",
    checkpoint_dir: Path | None = None,
    extra_callbacks: Sequence[L.Callback] = (),
    resume_checkpoint_path: str | None = None,
) -> L.Trainer:
    """Assemble the trainer from ``cfg`` and fit ``model``; return the trainer.

    The logger, the configured callbacks, the trainer and the hyperparameter log are built
    the same way for every stage that trains (``train``, QAT under ``quantize``).

    Args:
        cfg: Fully composed Hydra configuration.
        model: The model to train.
        datamodule: The data module.
        run_context: The MLflow run, or ``None`` without a tracker.
        config_name: User-facing config name, for the trainer logger.
        stage: Stage name recorded by the logger.
        checkpoint_dir: Where ``ModelCheckpoint`` writes; ``None`` keeps its configured path.
        extra_callbacks: Callbacks appended to the configured ones.
        resume_checkpoint_path: Checkpoint to resume from, if any.
    """
    logger_enabled = run_context is not None
    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=run_context,
        stage=stage,
        config_name=config_name,
        logger_enabled=logger_enabled,
    )
    callbacks = instantiate_callbacks(
        cfg, logger_enabled=logger_enabled, checkpoint_dir=checkpoint_dir
    )
    callbacks.extend(extra_callbacks)
    trainer_root_dir = (
        run_context.artifact_dir if run_context is not None else cfg.experiment_run_dir
    )
    trainer: L.Trainer = instantiate_trainer(cfg, callbacks, trainer_logger, trainer_root_dir)
    log_hyperparameters(cfg, trainer_logger)

    logger.info("Starting training...")
    logger.info(f"Max epochs: {cfg.trainer.max_epochs}")
    logger.info(f"Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    logger.info(f"Devices: {cfg.trainer.get('devices', 'auto')}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_checkpoint_path)
    logger.info("Training completed!")
    return trainer


def optimized_score(trainer: L.Trainer, cfg: DictConfig) -> float:
    """Return the metric a hyperparameter search optimizes (``optimized_metric``)."""
    optimized_metric = cfg.get("optimized_metric", "val/loss")
    score = trainer.callback_metrics.get(optimized_metric)
    if score is None:
        available_metrics = sorted(str(key) for key in trainer.callback_metrics)
        raise ValueError(
            f"Optimized metric '{optimized_metric}' was not logged. "
            f"Available callback metrics: {available_metrics}"
        )
    return float(score)


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    # First, so that every rank after rank zero stays silent from the config dump onwards.
    restrict_console_logging_to_rank_zero()
    log_configuration(cfg)
    config_name = HydraConfig.get().job.config_name
    if config_name is None:
        raise ValueError("Hydra config name is not available.")

    logger_enabled = cfg.get("logger") is not None
    config_name = config_name.removeprefix(CONFIG_NAME_PREFIX)

    run_context = build_mlflow_run_context(
        cfg,
        stage="train",
        config_name=config_name,
        logger_enabled=logger_enabled,
    )

    configure_torch_runtime()
    set_seed(cfg)

    system_metrics_monitor = None
    if run_context is not None and rank_zero_only.rank == 0:
        system_metrics_monitor = SystemMetricsMonitor(
            run_context.run_id, tracking_uri=run_context.tracking_uri
        )
        system_metrics_monitor.start()

    datamodule = build_datamodule(cfg)
    data_preprocessor = build_data_preprocessor(cfg)
    # A fine-tune config leaves weights at ??? and must be given --weights, so resolve it
    # rather than cfg.get, which reads ??? as None and would train from scratch. A resumed
    # run takes its weights from the checkpoint instead.
    resume_checkpoint_path = cfg.get("resume_checkpoint", None)
    weights_path = resolve_weights_paths(cfg, resuming=resume_checkpoint_path is not None)
    model = build_model(
        cfg,
        data_preprocessor=data_preprocessor,
        weights_path=weights_path,
        resume_checkpoint_path=resume_checkpoint_path,
        device=torch.device("cpu"),
        set_eval=False,
        enforce_full_coverage=False,
        skip_mismatched_weights=bool(cfg.get("skip_mismatched_weights", False)),
    )

    # Without a run context ModelCheckpoint keeps its configured ${hydra:run.dir}/checkpoints.
    checkpoint_dir = run_context.checkpoints_dir if run_context is not None else None
    trainer = fit_model(
        cfg,
        model,
        datamodule,
        run_context,
        config_name,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    logger.info(
        "Checkpoints saved to: %s",
        checkpoint_dir if checkpoint_dir is not None else resolve_work_dir() / "checkpoints",
    )
    score = optimized_score(trainer, cfg)
    if system_metrics_monitor is not None:
        system_metrics_monitor.finish()

    return score


if __name__ == "__main__":
    main()
