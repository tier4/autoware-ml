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

"""Reusable checkpoint-loading helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchingWeightsLoadReport:
    """Summary of a checkpoint-to-model matching-weight load."""

    loaded_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    not_loaded_model_keys: tuple[str, ...]


def load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a raw Lightning checkpoint payload from disk.

    Args:
        checkpoint_path: Path to the checkpoint file.
        map_location: Device or device string used when loading the checkpoint.

    Returns:
        Deserialized checkpoint payload.
    """
    return torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)


def _format_keys(keys: tuple[str, ...]) -> str:
    return ", ".join(keys) if keys else "<none>"


def _format_shape_mismatches(
    keys: tuple[str, ...],
    checkpoint_state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> str:
    if not keys:
        return "<none>"
    return ", ".join(
        f"{key}: checkpoint{tuple(checkpoint_state_dict[key].shape)} "
        f"!= model{tuple(model_state_dict[key].shape)}"
        for key in keys
    )


def load_matching_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    logger: logging.Logger | None = None,
) -> MatchingWeightsLoadReport:
    """Initialize matching model tensors from a Lightning checkpoint.

    Only tensors with the same state-dict key and shape are loaded. All other
    checkpoint tensors are reported and skipped. This is intended for model
    initialization from pretrained weights, not for training resume.

    Args:
        model: Target model instance.
        checkpoint_path: Lightning checkpoint path.
        map_location: Device or device string used when loading the checkpoint.
        logger: Logger used for the load report.

    Returns:
        Report containing loaded and skipped tensor keys.

    Raises:
        ValueError: If no checkpoint tensors match the target model.
    """
    active_logger = logger if logger is not None else _LOGGER
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    checkpoint_state_dict = checkpoint["state_dict"]
    model_state_dict = model.state_dict()

    unexpected_keys = tuple(
        sorted(key for key in checkpoint_state_dict if key not in model_state_dict)
    )
    shape_mismatched_keys = tuple(
        sorted(
            key
            for key, value in checkpoint_state_dict.items()
            if key in model_state_dict and value.shape != model_state_dict[key].shape
        )
    )
    if shape_mismatched_keys:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' contains {len(shape_mismatched_keys)} tensor(s) "
            "with matching names but incompatible shapes: "
            f"{_format_shape_mismatches(shape_mismatched_keys, checkpoint_state_dict, model_state_dict)}"
        )

    loaded_state_dict = {
        key: value
        for key, value in checkpoint_state_dict.items()
        if key in model_state_dict and value.shape == model_state_dict[key].shape
    }
    loaded_keys = tuple(sorted(loaded_state_dict))
    if not loaded_keys:
        raise ValueError(
            f"Weights checkpoint '{checkpoint_path}' does not contain any tensors matching "
            "the target model by key and shape."
        )

    not_loaded_model_keys = tuple(
        sorted(key for key in model_state_dict if key not in loaded_state_dict)
    )

    incompatible_keys = model.load_state_dict(loaded_state_dict, strict=False)
    if incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys were produced while loading pre-filtered matching weights: "
            f"{incompatible_keys.unexpected_keys}"
        )

    active_logger.info("Loaded weights from: %s", checkpoint_path)
    active_logger.info(
        "Loaded matching weight tensors: %d/%d", len(loaded_keys), len(model_state_dict)
    )
    active_logger.info("Loaded weight keys (%d): %s", len(loaded_keys), _format_keys(loaded_keys))
    active_logger.info(
        "Skipped checkpoint keys missing in model (%d): %s",
        len(unexpected_keys),
        _format_keys(unexpected_keys),
    )
    active_logger.info(
        "Model keys not initialized from weights (%d): %s",
        len(not_loaded_model_keys),
        _format_keys(not_loaded_model_keys),
    )

    return MatchingWeightsLoadReport(
        loaded_keys=loaded_keys,
        unexpected_keys=unexpected_keys,
        not_loaded_model_keys=not_loaded_model_keys,
    )


def apply_matching_weights(
    model: torch.nn.Module,
    weights: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    map_location: str | torch.device = "cpu",
    device: torch.device | None = None,
    set_eval: bool = False,
    enforce_full_coverage: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[MatchingWeightsLoadReport, ...]:
    """Apply one or more matching-weight checkpoints to a model.

    Args:
        model: Target model instance.
        weights: One checkpoint path or an ordered list of checkpoint paths.
        map_location: Device or device string used when loading each checkpoint.
        device: Optional device to move the model to after all weights are loaded.
        set_eval: Whether to switch the model into evaluation mode after loading.
        enforce_full_coverage: If true, raise when any model state_dict key was
            not loaded by any of the supplied checkpoints. Use this for deploy
            where every exported parameter must come from a trained checkpoint.
        logger: Logger used for per-checkpoint load reports.

    Returns:
        Per-checkpoint matching-weight load reports.

    Raises:
        RuntimeError: When ``enforce_full_coverage`` is true and one or more
            model parameters remain uncovered after all checkpoints are loaded.
    """
    weight_paths = (weights,) if isinstance(weights, (str, Path)) else tuple(weights)
    reports = tuple(
        load_matching_weights(
            model,
            Path(weight_path),
            map_location=map_location,
            logger=logger,
        )
        for weight_path in weight_paths
    )
    if enforce_full_coverage:
        loaded = set().union(*(report.loaded_keys for report in reports))
        missing = tuple(key for key in model.state_dict().keys() if key not in loaded)
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Model has {len(missing)} parameter(s) not covered by any checkpoint: "
                f"{joined}. Supply additional --weights or a checkpoint that includes these keys."
            )
    if device is not None:
        model.to(device)
    if set_eval:
        model.eval()
    return reports
