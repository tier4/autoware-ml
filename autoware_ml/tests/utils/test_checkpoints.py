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

"""Unit tests for evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import lightning as L
import torch

from autoware_ml.utils.checkpoints import (
    load_model_from_checkpoint,
    load_model_from_raw_checkpoint,
)


class _TinyModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)


def test_load_model_from_checkpoint_uses_full_checkpoint_loading(tmp_path: Path) -> None:
    model = _TinyModel()
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"placeholder")

    state_dict = model.state_dict()
    with patch(
        "autoware_ml.utils.checkpoints.torch.load", return_value={"state_dict": state_dict}
    ) as load_mock:
        load_model_from_checkpoint(model, checkpoint_path, map_location="cpu")

    load_mock.assert_called_once_with(str(checkpoint_path), map_location="cpu", weights_only=False)


def test_load_model_from_raw_checkpoint_uses_explicit_state_key(tmp_path: Path) -> None:
    model = _TinyModel()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint = {"ema": {"module": model.state_dict()}}
    torch.save(checkpoint, checkpoint_path)

    target = _TinyModel()
    incompatible = load_model_from_raw_checkpoint(
        target,
        checkpoint_path,
        state_key="ema.module",
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for key, value in model.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_load_model_from_raw_checkpoint_can_filter_mismatched_shapes(tmp_path: Path) -> None:
    source = _TinyModel()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint = {"model": source.state_dict()}
    torch.save(checkpoint, checkpoint_path)

    target = torch.nn.Linear(2, 2)
    incompatible = load_model_from_raw_checkpoint(
        target,
        checkpoint_path,
        filter_mismatched_shapes=True,
    )

    assert set(incompatible.missing_keys) == {"weight", "bias"}
    assert incompatible.unexpected_keys == []
