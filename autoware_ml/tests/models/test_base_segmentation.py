"""Tests for BaseSegmentationModel shared behavior."""

from __future__ import annotations

import torch
import torch.nn as nn

from autoware_ml.models.segmentation3d.base import BaseSegmentationModel


class _ConcreteSegModel(BaseSegmentationModel):
    """Minimal concrete segmentation model for testing."""

    def __init__(self, num_classes: int = 3, ignore_index: int = -1) -> None:
        super().__init__(optimizer=torch.optim.AdamW)
        self.head = nn.Linear(4, num_classes)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def compute_metrics(
        self, outputs: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        loss = nn.functional.cross_entropy(outputs, target, ignore_index=self.ignore_index)
        return {"loss": loss}

    def _get_point_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs


def test_predict_outputs_returns_labels_and_probs() -> None:
    model = _ConcreteSegModel()
    logits = torch.randn(5, 3)

    result = model.predict_outputs(logits)

    assert "pred_labels" in result
    assert "pred_probs" in result
    assert result["pred_labels"].shape == (5,)
    assert result["pred_probs"].shape == (5, 3)
    assert torch.allclose(result["pred_probs"].sum(dim=1), torch.ones(5))


def test_get_export_output_names_reads_class_attribute() -> None:
    model = _ConcreteSegModel()

    assert model.get_export_output_names() == ["pred_labels", "pred_probs"]


def test_subclass_can_override_export_output_names() -> None:
    class _SingleOutput(_ConcreteSegModel):
        EXPORT_OUTPUT_NAMES = ("pred_probs",)

    model = _SingleOutput()

    assert model.get_export_output_names() == ["pred_probs"]


def test_compute_segmentation_metrics_returns_expected_keys() -> None:
    model = _ConcreteSegModel(num_classes=3, ignore_index=-1)
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))

    metrics = model._compute_segmentation_metrics(logits, targets)

    assert "point_accuracy" in metrics
    assert "mean_iou" in metrics
    assert "mean_precision" in metrics
    assert "mean_recall" in metrics
    assert "mean_f1" in metrics


def test_compute_segmentation_metrics_returns_empty_when_all_ignored() -> None:
    model = _ConcreteSegModel(num_classes=3, ignore_index=-1)
    logits = torch.randn(5, 3)
    targets = torch.full((5,), -1, dtype=torch.long)

    metrics = model._compute_segmentation_metrics(logits, targets)

    assert metrics == {}
