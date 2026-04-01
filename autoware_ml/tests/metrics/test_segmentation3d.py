"""Tests for 3D semantic segmentation metrics."""

from __future__ import annotations

import torch

from autoware_ml.metrics.segmentation3d import (
    compute_point_accuracy,
    compute_segmentation_metrics,
)


def test_point_accuracy_skips_ignored_targets() -> None:
    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, -1, 1])

    accuracy = compute_point_accuracy(predictions, targets, ignore_index=-1)

    assert accuracy is not None
    assert torch.isclose(accuracy, torch.tensor(2.0 / 3.0))


def test_point_accuracy_returns_none_when_all_targets_are_ignored() -> None:
    predictions = torch.tensor([0, 1, 1])
    targets = torch.tensor([-1, -1, -1])

    accuracy = compute_point_accuracy(predictions, targets, ignore_index=-1)

    assert accuracy is None


def test_segmentation_metrics_returns_all_keys() -> None:
    predictions = torch.tensor([0, 1, 2, 1, 2, 0])
    targets = torch.tensor([0, 1, 2, 0, 2, -1])

    metrics = compute_segmentation_metrics(predictions, targets, num_classes=3, ignore_index=-1)

    assert metrics is not None
    assert set(metrics) == {
        "point_accuracy",
        "mean_iou",
        "mean_precision",
        "mean_recall",
        "mean_f1",
    }


def test_segmentation_metrics_values_match_hand_computation() -> None:
    predictions = torch.tensor([0, 1, 2, 1, 2, 0])
    targets = torch.tensor([0, 1, 2, 0, 2, -1])

    m = compute_segmentation_metrics(predictions, targets, num_classes=3, ignore_index=-1)

    assert torch.isclose(m["point_accuracy"], torch.tensor(4.0 / 5.0))
    assert torch.isclose(m["mean_iou"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(m["mean_precision"], torch.tensor(5.0 / 6.0))
    assert torch.isclose(m["mean_recall"], torch.tensor(5.0 / 6.0))
    assert torch.isclose(m["mean_f1"], torch.tensor(7.0 / 9.0))


def test_segmentation_metrics_returns_none_when_all_ignored() -> None:
    predictions = torch.tensor([0, 1])
    targets = torch.tensor([-1, -1])

    assert (
        compute_segmentation_metrics(predictions, targets, num_classes=2, ignore_index=-1) is None
    )


def test_segmentation_metrics_excludes_classes_without_support() -> None:
    """Class 2 has no ground-truth points - it must not affect the macro mean."""
    predictions = torch.tensor([0, 0, 1])
    targets = torch.tensor([0, 0, 1])

    m = compute_segmentation_metrics(predictions, targets, num_classes=3, ignore_index=-1)

    # Perfect predictions on classes 0 and 1; class 2 absent.
    assert torch.isclose(m["mean_iou"], torch.tensor(1.0))
    assert torch.isclose(m["mean_precision"], torch.tensor(1.0))
    assert torch.isclose(m["mean_recall"], torch.tensor(1.0))
    assert torch.isclose(m["mean_f1"], torch.tensor(1.0))
