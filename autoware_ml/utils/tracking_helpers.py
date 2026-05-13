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

"""Backend-neutral tracking helpers for runtime entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict

import autoware_ml.utils.mlflow_helpers as mlflow_helpers
import autoware_ml.utils.wandb_helpers as wandb_helpers

AUTOWARE_ML_HYDRA_RUN_DIR_ENV = mlflow_helpers.AUTOWARE_ML_HYDRA_RUN_DIR_ENV
AUTOWARE_ML_RUN_ID_ENV = mlflow_helpers.AUTOWARE_ML_RUN_ID_ENV

MLFLOW_TARGET = "lightning.pytorch.loggers.MLFlowLogger"
WANDB_TARGET = "lightning.pytorch.loggers.WandbLogger"


def get_tracking_backend(cfg: DictConfig) -> str | None:
    """Return the configured tracking backend name."""
    logger_cfg = cfg.get("logger")
    if logger_cfg is None:
        return None
    target = str(logger_cfg.get("_target_", ""))
    if target == MLFLOW_TARGET:
        return "mlflow"
    if target == WANDB_TARGET:
        return "wandb"
    return "custom"


def should_enable_logger(cfg: DictConfig) -> bool:
    """Return whether a trainer logger should be enabled."""
    return cfg.get("logger") is not None and not bool(cfg.trainer.get("fast_dev_run", False))


def get_user_config_name() -> str:
    """Return the current Hydra user-facing config name."""
    return mlflow_helpers.get_user_config_name()


def generate_hydra_run_dir(
    cfg: DictConfig | None,
    config_name: str,
    started_at=None,
) -> Path:
    """Return the fallback Hydra output directory for the selected backend."""
    if cfg is not None and get_tracking_backend(cfg) == "wandb":
        return wandb_helpers.generate_hydra_run_dir(
            config_name,
            save_dir=cfg.logger.get("save_dir", None),
            started_at=started_at,
        )
    return mlflow_helpers.generate_hydra_run_dir(config_name, started_at=started_at)


def resolve_lineage_context(
    cfg: DictConfig,
    config_name: str,
    checkpoint_path: Path | None,
) -> tuple[str, str | None]:
    """Resolve experiment/project lineage for the selected backend."""
    backend = get_tracking_backend(cfg)
    if backend == "wandb":
        return wandb_helpers.resolve_lineage_context(config_name, checkpoint_path)
    return mlflow_helpers.resolve_lineage_context(config_name, checkpoint_path)


def prepare_run_context(
    cfg: DictConfig,
    config_name: str,
    hydra_dir: Path | None,
    stage: str,
    parent_run_id: str | None = None,
    experiment_name: str | None = None,
    extra_tags: dict[str, Any] | None = None,
    started_at=None,
):
    """Create a run context for the selected tracking backend."""
    backend = get_tracking_backend(cfg)
    if backend == "wandb":
        return wandb_helpers.prepare_run_context(
            cfg.logger,
            config_name,
            hydra_dir=hydra_dir,
            stage=stage,
            parent_run_id=parent_run_id,
            extra_tags=extra_tags,
            started_at=started_at,
        )
    if backend == "mlflow":
        return mlflow_helpers.prepare_run_context(
            cfg.logger.tracking_uri,
            config_name,
            hydra_dir=hydra_dir,
            stage=stage,
            parent_run_id=parent_run_id,
            experiment_name=experiment_name,
            extra_tags=extra_tags,
            started_at=started_at,
        )
    raise ValueError(f"Unsupported logger backend: {backend}")


def load_run_context(
    cfg: DictConfig,
    run_id: str,
    hydra_dir: Path | None = None,
    config_name: str | None = None,
    stage: str = "train",
):
    """Load a pre-created run context for the selected backend."""
    backend = get_tracking_backend(cfg)
    if backend == "wandb":
        with open_dict(cfg.logger):
            cfg.logger.id = run_id
            cfg.logger.version = run_id
        return wandb_helpers.prepare_run_context(
            cfg.logger,
            config_name or str(cfg.logger.get("group", "autoware-ml")),
            hydra_dir=hydra_dir,
            stage=stage,
        )
    if backend == "mlflow":
        return mlflow_helpers.load_run_context(cfg.logger.tracking_uri, run_id)
    raise ValueError(f"Unsupported logger backend: {backend}")


def configure_logger(cfg: DictConfig, run_context) -> None:
    """Populate the configured Lightning logger node from a run context."""
    backend = get_tracking_backend(cfg)
    if backend == "wandb":
        wandb_helpers.configure_logger(cfg.logger, run_context)
        return
    if backend == "mlflow":
        mlflow_helpers.configure_logger(
            cfg.logger,
            run_context.experiment_name,
            run_context.run_name,
            run_context.tags,
            run_id=run_context.run_id,
        )
        return
    raise ValueError(f"Unsupported logger backend: {backend}")


def write_run_config_artifacts(cfg: DictConfig, run_context) -> None:
    """Persist config artifacts for the selected backend."""
    if get_tracking_backend(cfg) == "wandb":
        wandb_helpers.write_run_config_artifacts(cfg, run_context.artifact_dir)
    else:
        mlflow_helpers.write_run_config_artifacts(cfg, run_context.artifact_dir)


def build_run_metadata(
    cfg: DictConfig,
    run_context,
    config_name: str,
    hydra_dir: Path,
    stage: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build persisted run metadata for the selected backend."""
    if get_tracking_backend(cfg) == "wandb":
        return wandb_helpers.build_run_metadata(
            run_context,
            config_name,
            hydra_dir,
            stage,
            extra_metadata=extra_metadata,
        )
    return mlflow_helpers.build_run_metadata(
        run_context,
        config_name,
        hydra_dir,
        stage,
        extra_metadata=extra_metadata,
    )


def write_run_metadata(cfg: DictConfig, run_context, metadata: dict[str, Any]) -> Path:
    """Persist run metadata for the selected backend."""
    if get_tracking_backend(cfg) == "wandb":
        return wandb_helpers.write_run_metadata(run_context.artifact_dir, metadata)
    return mlflow_helpers.write_run_metadata(run_context.artifact_dir, metadata)


def log_stage_artifacts(cfg: DictConfig, trainer_logger, run_context, stage: str) -> None:
    """Log common post-stage artifacts for backends that need explicit uploads."""
    if get_tracking_backend(cfg) != "wandb" or trainer_logger is None:
        return
    wandb_helpers.log_stage_artifacts(trainer_logger.experiment, run_context, stage)
