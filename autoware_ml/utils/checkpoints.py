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

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _resolve_state_dict(
    checkpoint: Mapping[str, Any] | Mapping[str, torch.Tensor],
    state_key: str | None = None,
) -> Mapping[str, torch.Tensor]:
    """Resolve a tensor state dict from a generic checkpoint object."""

    if state_key is not None:
        current: Any = checkpoint
        for key_part in state_key.split("."):
            if not isinstance(current, Mapping) or key_part not in current:
                raise KeyError(f"Checkpoint state key '{state_key}' could not be resolved.")
            current = current[key_part]
        if not isinstance(current, Mapping):
            raise ValueError(f"Resolved state key '{state_key}' is not a mapping.")
        return current

    for key_path in (("state_dict",), ("model",)):
        current: Any = checkpoint
        found = True
        for key_part in key_path:
            if not isinstance(current, Mapping) or key_part not in current:
                found = False
                break
            current = current[key_part]
        if found and isinstance(current, Mapping):
            return current

    if isinstance(checkpoint, Mapping):
        return checkpoint

    raise ValueError("Could not resolve a tensor state dict from the checkpoint.")


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
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if device is not None:
        model.to(device)
    if set_eval:
        model.eval()


def load_model_from_raw_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    state_key: str | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    filter_mismatched_shapes: bool = False,
    device: torch.device | None = None,
    set_eval: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load weights from a non-Lightning checkpoint.

    The loader supports raw state dicts as well as common nested checkpoint
    layouts such as ``state_dict`` and ``model``. For less common nested
    layouts like ``ema.module``, pass ``state_key`` explicitly.
    """

    checkpoint = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    state_dict = _resolve_state_dict(checkpoint, state_key=state_key)

    if filter_mismatched_shapes:
        model_state_dict = model.state_dict()
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state_dict and tuple(value.shape) == tuple(model_state_dict[key].shape)
        }
        strict = False

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if device is not None:
        model.to(device)
    if set_eval:
        model.eval()
    return incompatible
