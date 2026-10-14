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

import logging
from pathlib import Path
from typing import Sequence

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.checkpoints import apply_matching_weights

logger = logging.getLogger(__name__)


def build_data_preprocessor(cfg: DictConfig) -> DataPreprocessor:
    """
    Build a data preprocessor from the Hydra configuration.

    Args:
        cfg: Hydra configuration.
    """
    logger.info("Building data preprocessor...")
    data_preprocessor = instantiate(cfg.data_preprocessor)
    logger.info(f"Data preprocessor built successfully with {data_preprocessor}.")
    return data_preprocessor


def build_model(
    cfg: DictConfig,
    data_preprocessor: DataPreprocessor,
    device: torch.device,
    weights_path: str | Path | Sequence[str | Path] | tuple[str | Path, ...] | None,
    resume_checkpoint_path: str | None,
    set_eval: bool = False,
    enforce_full_coverage: bool = False,
    skip_mismatched_weights: bool = False,
) -> ModuleBaseModel:
    """
    Build a model from the Hydra configuration.

    Args:
        cfg: Hydra configuration.
        data_preprocessor: Data preprocessor to be used with the model.
        device: Device to load the model onto.
        weights_path: Path(s) to the weights file(s) to load into the model.
        resume_checkpoint_path: Path to the checkpoint file to resume training from.
        set_eval: Whether to set the model to evaluation mode after loading weights.
        enforce_full_coverage: Whether to enforce that all model parameters are covered by the weights.
        skip_mismatched_weights: Whether to skip, with a warning, weight tensors whose key matches a
            model tensor of a different shape instead of raising.

    Returns:
        Pytorch-Lightning ModuleBaseModel for multi-task learning/inference.
    """
    logger.info("Building model...")
    model = instantiate(cfg.model, data_preprocessor=data_preprocessor)

    # Resume checkpoint
    if resume_checkpoint_path is not None and weights_path is not None:
        raise ValueError("'--resume-checkpoint' and '--weights' are mutually exclusive.")

    if weights_path is not None:
        apply_matching_weights(
            model,
            weights_path,
            map_location=device,
            logger=logger,
            enforce_full_coverage=enforce_full_coverage,
            skip_mismatched_shapes=skip_mismatched_weights,
            set_eval=set_eval,
        )

    if resume_checkpoint_path is not None:
        progress = torch.load(
            resume_checkpoint_path, map_location=device, weights_only=False, mmap=True
        )
        logger.info(
            "Resuming from '%s': checkpoint saved at epoch %d (global step %d), "
            "training continues at epoch %d.",
            resume_checkpoint_path,
            progress["epoch"],
            progress["global_step"],
            progress["epoch"] + 1,
        )
    logger.info(f"Model built successfully with {model}.")
    return model


def build_weight_checkpoint_paths(cfg: DictConfig) -> tuple[Sequence[Path], Path]:
    """
    Build a list of weight checkpoint paths from the Hydra configuration.

    Args:
        cfg: Hydra configuration.
    Returns:
        Tuple containing a list of weight checkpoint paths and the last checkpoint path.
    """
    weights_arg = cfg.get("weights", None)
    if weights_arg is None:
        raise ValueError("--weights <path> (repeatable) must be specified.")

    weight_paths = (
        [Path(weights_arg)]
        if isinstance(weights_arg, str)
        else [Path(path) for path in weights_arg]
    )
    checkpoint_path = Path(weight_paths[-1])
    for path in weight_paths:
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

    return weight_paths, checkpoint_path
