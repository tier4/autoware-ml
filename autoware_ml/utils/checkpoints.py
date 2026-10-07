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


# --- Resume helpers for preemptible runs ----------------------------------------------------

AUTOWARE_ML_RUN_POINTER_FILE_ENV = "AUTOWARE_ML_RUN_POINTER_FILE"
"""Environment variable naming a file that receives the run's checkpoint directory.

A slurm launch script sets it to a path derived from the stable job id; the training
entrypoint writes the checkpoint directory of its run there, and the relaunch after a
requeue reads it back to pass ``--resume-latest``.
"""

_RESUME_REQUIRED_KEYS = ("state_dict", "optimizer_states", "loops")


def write_run_pointer(
    checkpoints_dir: Path, pointer_env: str = AUTOWARE_ML_RUN_POINTER_FILE_ENV
) -> None:
    """Write the checkpoint directory of the current run to the pointer file, if requested.

    Args:
        checkpoints_dir: Directory the run's checkpoints are written to.
        pointer_env: Environment variable holding the pointer file path.
    """
    import os

    pointer = os.environ.get(pointer_env)
    if not pointer:
        return
    pointer_path = Path(pointer)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    tmp_path.write_text(f"{checkpoints_dir}\n")
    os.replace(tmp_path, pointer_path)
    _LOGGER.info("Wrote the run pointer '%s' -> '%s'.", pointer_path, checkpoints_dir)


def checkpoint_is_loadable(checkpoint_path: Path) -> bool:
    """Return whether a checkpoint file holds a complete, readable training state.

    A job killed while a file was being written can leave a truncated checkpoint
    behind; this reads it through and checks the keys a resume needs.

    Args:
        checkpoint_path: Candidate checkpoint file.

    Returns:
        ``True`` when the file loads and carries the training state.
    """
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as error:  # noqa: BLE001 - any read error means "not this file"
        _LOGGER.warning("Checkpoint '%s' is not loadable (%s).", checkpoint_path, error)
        return False
    missing = [key for key in _RESUME_REQUIRED_KEYS if key not in payload]
    if missing:
        _LOGGER.warning("Checkpoint '%s' misses %s; skipping it.", checkpoint_path, missing)
        return False
    return True


def find_latest_checkpoint(checkpoints_dir: Path, last_name: str = "last.ckpt") -> Path | None:
    """Return the newest loadable checkpoint in a directory, or ``None``.

    ``last.ckpt`` is tried first when it exists (following a symlink), then every
    other ``*.ckpt`` file from the newest to the oldest by modification time. Files
    that do not load are skipped, so a checkpoint truncated by a kill never blocks
    the resume.

    Args:
        checkpoints_dir: Directory to search.
        last_name: Name of the pointer checkpoint written by the periodic callback.

    Returns:
        The checkpoint to resume from, or ``None`` when the directory holds none.
    """
    if not checkpoints_dir.is_dir():
        return None
    candidates: list[Path] = []
    last = checkpoints_dir / last_name
    if last.exists():
        candidates.append(last.resolve())
    others = [
        path.resolve()
        for path in checkpoints_dir.glob("*.ckpt")
        if path.name != last_name and path.is_file()
    ]
    others.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in others:
        if path not in candidates:
            candidates.append(path)
    for candidate in candidates:
        if checkpoint_is_loadable(candidate):
            return candidate
    return None
