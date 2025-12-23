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

"""Training script for Autoware-ML models."""

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import autoware_ml.configs

logger = logging.getLogger(__name__)


def generate_experiment_name() -> str:
    """Generate experiment name from config path."""
    config_name = HydraConfig.get().job.config_name
    experiment_name = config_name.replace("tasks/", "").replace("/", "_")
    return experiment_name


_CONFIG_PATH = str(Path(autoware_ml.configs.__file__).parent.resolve())


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

    # Get Hydra's output directory
    hydra_cfg = HydraConfig.get()
    work_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"Working directory: {work_dir}")

    if torch.cuda.is_available():
        # Set TF32 precision for performance on Ampere+ GPUs
        # Use legacy API for compatibility with PyTorch Lightning
        # Refactor after https://github.com/Lightning-AI/pytorch-lightning/pull/21306 is merged
        torch.set_float32_matmul_precision("medium")  # TF32 for matmul
        torch.backends.cudnn.fp32_precision = "tf32"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info("TF32 precision enabled for improved performance")

    # Set random seed
    if "seed" in cfg:
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Random seed: {cfg.seed}")

    logger.info("Instantiating datamodule...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info("Instantiating model...")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks = []
    if "callbacks" in cfg:
        for callback_cfg in cfg.callbacks.values():
            callbacks.append(hydra.utils.instantiate(callback_cfg))

    logger.info("Instantiating loggers...")
    trainer_logger = None
    exp_name = generate_experiment_name()
    if "logger" in cfg:
        with open_dict(cfg.logger):
            cfg.logger.experiment_name = exp_name
        trainer_logger = hydra.utils.instantiate(cfg.logger)

    logger.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=trainer_logger,
        default_root_dir=str(work_dir),
    )

    # Log hyperparameters
    if trainer_logger:
        hparams = OmegaConf.to_container(cfg, resolve=True)
        trainer.logger.log_hyperparams(hparams)

    # Start training
    logger.info("Starting training...")
    logger.info(f"Max epochs: {cfg.trainer.max_epochs}")
    logger.info(f"Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    logger.info(f"Devices: {cfg.trainer.get('devices', 'auto')}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint", None))

    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {work_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
