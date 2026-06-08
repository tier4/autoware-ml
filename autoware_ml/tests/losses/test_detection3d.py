"""Tests for detection3d loss functions."""

from __future__ import annotations

import torch

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss


def test_sigmoid_focal_loss_clamps_avg_factor() -> None:
    loss_fn = SigmoidFocalLoss()
    logits = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    unclamped = loss_fn(logits, targets)
    clamped = loss_fn(logits, targets, avg_factor=0.0)

    assert torch.isclose(clamped, unclamped)


def test_sigmoid_focal_loss_broadcasts_query_weights() -> None:
    loss_fn = SigmoidFocalLoss()
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    weights = torch.tensor([1.0, 0.0], dtype=torch.float32)

    weighted = loss_fn(logits, targets, weights=weights)
    expected = loss_fn(logits[:1], targets[:1])

    assert torch.isclose(weighted, expected)


def test_gaussian_focal_loss_handles_zero_positive_heatmap() -> None:
    loss_fn = GaussianFocalLoss()
    prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    target = torch.zeros_like(prediction)

    loss = loss_fn(prediction, target)

    assert torch.isfinite(loss)
    assert loss > 0
