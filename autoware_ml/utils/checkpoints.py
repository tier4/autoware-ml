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

from pathlib import Path

import torch


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
