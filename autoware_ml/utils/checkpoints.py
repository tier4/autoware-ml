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
    shape_mismatched_keys: tuple[str, ...]
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


def extract_prefixed_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Extract one prefixed submodule state dict from a model state dict.

    Args:
        state_dict: Flat model state dict.
        prefix: Prefix identifying the submodule.

    Returns:
        Dictionary containing only keys that start with ``prefix``.
    """
    return {key: value for key, value in state_dict.items() if key.startswith(prefix)}


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
        "Skipped checkpoint keys with shape mismatch (%d): %s",
        len(shape_mismatched_keys),
        _format_shape_mismatches(shape_mismatched_keys, checkpoint_state_dict, model_state_dict),
    )
    active_logger.info(
        "Model keys not initialized from weights (%d): %s",
        len(not_loaded_model_keys),
        _format_keys(not_loaded_model_keys),
    )

    return MatchingWeightsLoadReport(
        loaded_keys=loaded_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatched_keys=shape_mismatched_keys,
        not_loaded_model_keys=not_loaded_model_keys,
    )


def apply_matching_weights(
    model: torch.nn.Module,
    weights: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    map_location: str | torch.device = "cpu",
    device: torch.device | None = None,
    set_eval: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[MatchingWeightsLoadReport, ...]:
    """Apply one or more matching-weight checkpoints to a model.

    Args:
        model: Target model instance.
        weights: One checkpoint path or an ordered list of checkpoint paths.
        map_location: Device or device string used when loading each checkpoint.
        device: Optional device to move the model to after all weights are loaded.
        set_eval: Whether to switch the model into evaluation mode after loading.
        logger: Logger used for per-checkpoint load reports.

    Returns:
        Per-checkpoint matching-weight load reports.
    """
    weight_paths = (weights,) if isinstance(weights, str | Path) else tuple(weights)
    reports = tuple(
        load_matching_weights(
            model,
            Path(weight_path),
            map_location=map_location,
            logger=logger,
        )
        for weight_path in weight_paths
    )
    if device is not None:
        model.to(device)
    if set_eval:
        model.eval()
    return reports


def assert_checkpoint_recipe_source(
    checkpoint_path: Path,
    expected_source_checkpoint_path: Path,
    *,
    recipe_type: str,
    source_key: str,
    checkpoint_label: str,
    source_label: str,
) -> None:
    """Require checkpoint metadata to reference the expected source checkpoint.

    Args:
        checkpoint_path: Checkpoint carrying ``autoware_ml_checkpoint_recipe`` metadata.
        expected_source_checkpoint_path: Source checkpoint expected by the current config.
        recipe_type: Required recipe type.
        source_key: Metadata key containing the source checkpoint path.
        checkpoint_label: Human-readable label for the checkpoint being checked.
        source_label: Human-readable label for the expected source checkpoint.

    Raises:
        RuntimeError: If the recipe is missing or references a different source.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    recipe = checkpoint.get("autoware_ml_checkpoint_recipe")
    if not isinstance(recipe, dict):
        raise RuntimeError(
            f"{checkpoint_label} '{checkpoint_path}' does not contain "
            "'autoware_ml_checkpoint_recipe' metadata."
        )
    if recipe.get("type") != recipe_type:
        raise RuntimeError(
            f"{checkpoint_label} '{checkpoint_path}' has recipe type "
            f"{recipe.get('type')!r}, expected {recipe_type!r}."
        )

    actual_source = recipe.get(source_key)
    if not isinstance(actual_source, str) or not actual_source:
        raise RuntimeError(
            f"{checkpoint_label} '{checkpoint_path}' recipe does not define '{source_key}'."
        )

    actual_source_path = Path(actual_source)
    actual_matches_expected = (
        actual_source_path.as_posix() == expected_source_checkpoint_path.as_posix()
    )
    if actual_source_path.exists() and expected_source_checkpoint_path.exists():
        actual_matches_expected = (
            actual_source_path.resolve() == expected_source_checkpoint_path.resolve()
        )
    if actual_matches_expected:
        return

    raise RuntimeError(
        f"{checkpoint_label} '{checkpoint_path}' was trained from {source_label} "
        f"'{actual_source}', but the current config uses "
        f"'{expected_source_checkpoint_path}'. Use the exact source checkpoint."
    )


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    device: torch.device | None = None,
    set_eval: bool = False,
) -> None:
    """Load model weights from a Lightning checkpoint.

    Args:
        model: Module instance restored from the checkpoint.
        checkpoint_path: Path to the checkpoint file.
        map_location: Device or device string used when loading the checkpoint.
        strict: Whether to enforce strict ``state_dict`` key matching.
        device: Optional device to move the model to after loading.
        set_eval: Whether to switch the model into evaluation mode after loading.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if device is not None:
        model.to(device)
    if set_eval:
        model.eval()


def load_submodule_from_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: Path,
    *,
    prefix: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> None:
    """Load one named submodule from a Lightning checkpoint.

    Args:
        module: Target submodule instance restored from the checkpoint.
        checkpoint_path: Path to the checkpoint file.
        prefix: State-dict key prefix identifying the serialized submodule.
        map_location: Device or device string used when loading the checkpoint.
        strict: Whether to enforce strict key matching on the submodule.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["state_dict"]
    module_state_dict = {
        key.removeprefix(prefix): value
        for key, value in extract_prefixed_state_dict(state_dict, prefix).items()
    }
    if not module_state_dict:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' does not contain any weights with prefix '{prefix}'."
        )
    module.load_state_dict(module_state_dict, strict=strict)
