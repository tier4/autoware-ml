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

"""Shared runtime helpers for train/test entrypoints.

This module centralizes setup logic used by runtime entrypoints, including
seeding, callback creation, trainer construction, and logging helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import autoware_ml.configs
import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    """Return the bundled Hydra config root path.

    Returns:
        Absolute path to the ``autoware_ml.configs`` package directory.
    """
    return str(Path(autoware_ml.configs.__file__).parent.resolve())


def log_configuration(cfg: DictConfig) -> None:
    """Log the resolved Hydra configuration.

    Args:
        cfg: Fully composed Hydra configuration.
    """
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)


def resolve_work_dir() -> Path:
    """Return Hydra's runtime output directory.

    Returns:
        Output directory assigned to the current Hydra job.
    """
    return Path(HydraConfig.get().runtime.output_dir)


def configure_torch_runtime() -> None:
    """Apply shared PyTorch runtime settings for GPU execution.

    The helper enables TF32-friendly settings when CUDA is available and logs
    the active GPU device.
    """
    if not torch.cuda.is_available():
        return

    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("TF32 precision enabled for improved performance")


def set_seed(cfg: DictConfig) -> None:
    """Seed all supported random generators when configured.

    Args:
        cfg: Fully composed Hydra configuration.
    """
    if "seed" not in cfg:
        return
    L.seed_everything(cfg.seed, workers=True)
    logger.info(f"Random seed: {cfg.seed}")


def instantiate_callbacks(cfg: DictConfig, logger_enabled: bool = True) -> list[L.Callback]:
    """Instantiate configured Lightning callbacks.

    Args:
        cfg: Fully composed Hydra configuration.
        logger_enabled: Whether the trainer will attach a logger.

    Returns:
        Instantiated callback list. Returns an empty list when callbacks are
        not configured.
    """
    if "callbacks" not in cfg:
        return []

    callbacks = []
    for callback_cfg in cfg.callbacks.values():
        if callback_cfg is None:
            continue
        target = callback_cfg.get("_target_", "")
        if not logger_enabled and target == "lightning.pytorch.callbacks.LearningRateMonitor":
            continue
        callbacks.append(hydra.utils.instantiate(callback_cfg))
    return callbacks


def instantiate_trainer(
    cfg: DictConfig,
    callbacks: list[L.Callback],
    trainer_logger: Logger | None,
    work_dir: Path,
) -> L.Trainer:
    """Instantiate the Lightning trainer for the current stage.

    Args:
        cfg: Fully composed Hydra configuration.
        callbacks: Callback instances attached to the trainer.
        trainer_logger: Optional logger instance passed to the trainer.
        work_dir: Runtime output directory used as the trainer root.

    Returns:
        Instantiated Lightning trainer.
    """
    return hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=trainer_logger if trainer_logger is not None else False,
        default_root_dir=str(work_dir),
    )


def log_hyperparameters(cfg: DictConfig, trainer_logger: Logger | None, trainer: L.Trainer) -> None:
    """Log resolved hyperparameters through the configured trainer logger.

    Args:
        cfg: Fully composed Hydra configuration.
        trainer_logger: Logger configured for the current trainer, if any.
        trainer: Instantiated Lightning trainer.
    """
    if trainer_logger is None:
        return
    trainer.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
