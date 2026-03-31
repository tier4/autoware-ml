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

from autoware_ml.utils.checkpoints import load_model_from_checkpoint


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
