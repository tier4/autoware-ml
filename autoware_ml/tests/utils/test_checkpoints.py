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

"""Unit tests for evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch

from autoware_ml.utils.checkpoints import (
    apply_matching_weights,
    load_matching_weights,
)


class _TinyModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)


def test_load_matching_weights_loads_same_key_and_shape_tensors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    model = _TinyModel()
    checkpoint_path = tmp_path / "weights.ckpt"
    checkpoint_state_dict = {
        "layer.weight": torch.full_like(model.layer.weight, 3.0),
        "extra.weight": torch.ones(1),
    }
    torch.save({"state_dict": checkpoint_state_dict}, checkpoint_path)

    report = load_matching_weights(model, checkpoint_path)

    assert report.loaded_keys == ("layer.weight",)
    assert report.unexpected_keys == ("extra.weight",)
    assert report.not_loaded_model_keys == ("layer.bias",)
    assert torch.equal(model.layer.weight, torch.full_like(model.layer.weight, 3.0))
    assert "Loaded weight keys (1): layer.weight" in caplog.text
    assert "Skipped checkpoint keys missing in model (1): extra.weight" in caplog.text
    assert "Model keys not initialized from weights (1): layer.bias" in caplog.text


def test_load_matching_weights_raises_on_shape_mismatch(tmp_path: Path) -> None:
    model = _TinyModel()
    checkpoint_path = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {"layer.bias": torch.ones(2)}}, checkpoint_path)

    with pytest.raises(ValueError, match="incompatible shapes"):
        load_matching_weights(model, checkpoint_path)


def test_load_matching_weights_rejects_checkpoints_without_matching_tensors(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    checkpoint_path = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {"other.weight": torch.ones(1)}}, checkpoint_path)

    with pytest.raises(ValueError, match="does not contain any tensors matching"):
        load_matching_weights(model, checkpoint_path)


def test_apply_matching_weights_loads_ordered_checkpoint_sequence(tmp_path: Path) -> None:
    model = _TinyModel()
    first_checkpoint_path = tmp_path / "first.ckpt"
    second_checkpoint_path = tmp_path / "second.ckpt"
    torch.save(
        {"state_dict": {"layer.weight": torch.full_like(model.layer.weight, 2.0)}},
        first_checkpoint_path,
    )
    torch.save(
        {"state_dict": {"layer.weight": torch.full_like(model.layer.weight, 4.0)}},
        second_checkpoint_path,
    )

    reports = apply_matching_weights(model, (first_checkpoint_path, second_checkpoint_path))

    assert len(reports) == 2
    assert reports[0].loaded_keys == ("layer.weight",)
    assert reports[1].loaded_keys == ("layer.weight",)
    assert torch.equal(model.layer.weight, torch.full_like(model.layer.weight, 4.0))


def test_apply_matching_weights_accepts_single_path_string(tmp_path: Path) -> None:
    model = _TinyModel()
    checkpoint_path = tmp_path / "weights.ckpt"
    torch.save(
        {"state_dict": {"layer.weight": torch.full_like(model.layer.weight, 5.0)}},
        checkpoint_path,
    )

    reports = apply_matching_weights(model, str(checkpoint_path))

    assert len(reports) == 1
    assert reports[0].loaded_keys == ("layer.weight",)
    assert torch.equal(model.layer.weight, torch.full_like(model.layer.weight, 5.0))


def test_apply_matching_weights_full_coverage_passes_when_union_complete(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    weight_checkpoint = tmp_path / "weights.ckpt"
    bias_checkpoint = tmp_path / "bias.ckpt"
    torch.save(
        {"state_dict": {"layer.weight": torch.full_like(model.layer.weight, 2.0)}},
        weight_checkpoint,
    )
    torch.save(
        {"state_dict": {"layer.bias": torch.full_like(model.layer.bias, 7.0)}},
        bias_checkpoint,
    )

    reports = apply_matching_weights(
        model, (weight_checkpoint, bias_checkpoint), enforce_full_coverage=True
    )

    assert len(reports) == 2
    assert torch.equal(model.layer.weight, torch.full_like(model.layer.weight, 2.0))
    assert torch.equal(model.layer.bias, torch.full_like(model.layer.bias, 7.0))


def test_apply_matching_weights_full_coverage_raises_when_keys_missing(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    weight_only_checkpoint = tmp_path / "weights.ckpt"
    torch.save(
        {"state_dict": {"layer.weight": torch.full_like(model.layer.weight, 3.0)}},
        weight_only_checkpoint,
    )

    with pytest.raises(RuntimeError, match="layer.bias"):
        apply_matching_weights(model, weight_only_checkpoint, enforce_full_coverage=True)
