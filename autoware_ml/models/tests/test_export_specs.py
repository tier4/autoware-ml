"""Framework export-contract tests for BaseModel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from autoware_ml.models.base import BaseModel


class _StructuredExportModel(BaseModel):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"sum": x + y, "diff": x - y}

    def compute_metrics(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict
        return {"loss": outputs["sum"].sum()}

    def predict_outputs(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict
        return outputs

    def get_export_output_names(self) -> list[str]:
        return ["sum", "diff"]


class _BatchExportModel(BaseModel):
    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch_inputs_dict["x"] + 1.0

    def compute_metrics(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict
        return {"loss": outputs.sum()}


class _UnnamedStructuredModel(_StructuredExportModel):
    def get_export_output_names(self) -> list[str] | None:
        return None


def test_build_export_spec_uses_forward_signature_inputs() -> None:
    model = _StructuredExportModel()
    batch = {
        "x": torch.tensor([2.0]),
        "y": torch.tensor([0.5]),
        "unused": torch.tensor([99.0]),
    }

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == ["x", "y"]
    assert spec.output_names == ["sum", "diff"]
    assert len(spec.args) == 2
    assert isinstance(outputs, tuple)
    assert torch.equal(outputs[0], torch.tensor([2.5]))
    assert torch.equal(outputs[1], torch.tensor([1.5]))


def test_build_export_spec_supports_whole_batch_forward() -> None:
    model = _BatchExportModel()
    batch = {"x": torch.tensor([2.0])}

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == ["batch_inputs_dict"]
    assert spec.args == (batch,)
    assert torch.equal(outputs, torch.tensor([3.0]))


def test_structured_export_outputs_require_names() -> None:
    model = _UnnamedStructuredModel()

    with pytest.raises(ValueError, match="explicit export output names"):
        model.prepare_export_outputs({"sum": torch.tensor([1.0])})
