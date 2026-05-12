"""Framework prediction-step contracts for BaseModel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from autoware_ml.models.base import BaseModel
from autoware_ml.preprocessing.base import DataPreprocessing


class _AddPreprocessedFeature:
    def __call__(self, batch_inputs_dict: dict[str, Any]) -> dict[str, Any]:
        return {"x": batch_inputs_dict["x"] + 1.0, "preprocessed": torch.tensor(True)}


class _ToyModel(BaseModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    def compute_metrics(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict
        return {"loss": outputs.sum()}

    def predict_outputs(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        return {
            "prediction": outputs,
            "preprocessed": batch_inputs_dict["preprocessed"],
        }


class _MissingLossModel(_ToyModel):
    def compute_metrics(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict, outputs
        return {"accuracy": torch.tensor(1.0)}


def test_on_after_batch_transfer_applies_preprocessing_pipeline() -> None:
    model = _ToyModel()
    model.set_data_preprocessing(DataPreprocessing([_AddPreprocessedFeature()]))

    batch = model.on_after_batch_transfer({"x": torch.tensor([1.0])}, dataloader_idx=0)

    assert torch.equal(batch["x"], torch.tensor([2.0]))
    assert batch["preprocessed"].item() is True


def test_predict_step_runs_forward_and_formats_predictions() -> None:
    model = _ToyModel()
    model.set_data_preprocessing(DataPreprocessing([_AddPreprocessedFeature()]))

    batch = model.on_after_batch_transfer({"x": torch.tensor([1.0])}, dataloader_idx=0)
    predictions = model.predict_step(batch, batch_idx=0)

    assert torch.equal(predictions["prediction"], torch.tensor([4.0]))
    assert predictions["preprocessed"].item() is True


def test_shared_step_requires_loss_metric() -> None:
    model = _MissingLossModel()

    with pytest.raises(ValueError, match="'loss' key"):
        model.training_step({"x": torch.tensor([1.0])}, batch_idx=0)
