"""Tests for 3D semantic segmentation metrics."""

from __future__ import annotations

import torch
import pytest

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.pointwise import (
    compute_point_accuracy,
    compute_segmentation_metrics,
)
from autoware_ml.metrics.segmentation3d.precision_recall_f1 import PrecisionRecallF1
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DMetricSuite


def _suite(**kwargs):
    """Build a segmentation suite with the full standard metric set injected."""
    return Segmentation3DMetricSuite(
        components=[IoU(), Accuracy(), PrecisionRecallF1()],
        **kwargs,
    )


def _coord(num_points: int) -> torch.Tensor:
    """Origin-centered point coordinates for tests where range does not matter."""
    return torch.zeros(num_points, 3)


def test_autoware_seg3d_metric_validation_reports_headline_scalars() -> None:
    metric = _suite(num_classes=3, ignore_index=-1)
    metric.update(
        {
            "seg_pred_labels": torch.tensor([0, 0, 1, 1, 2, -1]),
            "seg_target_labels": torch.tensor([0, 0, 1, 1, 2, -1]),
            "seg_coord": _coord(6),
        }
    )

    val = metric.result(EvalStage.VAL)

    # Validation reports exactly the three headline scalars, nothing per class.
    assert set(val) == {"mIoU", "acc", "mRecall"}
    assert val["mIoU"] == pytest.approx(1.0)
    assert val["acc"] == pytest.approx(1.0)
    assert val["mRecall"] == pytest.approx(1.0)


def test_autoware_seg3d_metric_test_reports_full_and_per_class() -> None:
    metric = _suite(num_classes=3, ignore_index=-1)
    metric.update(
        {
            "seg_pred_labels": torch.tensor([0, 0, 1, 1, 2, -1]),
            "seg_target_labels": torch.tensor([0, 0, 1, 1, 2, -1]),
            "seg_coord": _coord(6),
        }
    )

    test = metric.result(EvalStage.TEST)

    for scalar in ("mIoU", "acc", "mRecall", "fwIoU", "mPrecision", "mF1"):
        assert test[scalar] == pytest.approx(1.0)
    for class_index in range(3):
        for family in ("iou", "recall", "precision", "f1"):
            assert test[f"{family}_class_{class_index}"] == pytest.approx(1.0)


def test_autoware_seg3d_metric_excludes_classes_without_support() -> None:
    # Class 2 never appears in the ground truth, so it must not be reported.
    metric = _suite(num_classes=3, ignore_index=-1)
    metric.update(
        {
            "seg_pred_labels": torch.tensor([0, 1]),
            "seg_target_labels": torch.tensor([0, 1]),
            "seg_coord": _coord(2),
        }
    )

    test = metric.result(EvalStage.TEST)

    assert "iou_class_2" not in test
    assert "iou_class_0" in test and "iou_class_1" in test


def test_autoware_seg3d_metric_per_class_values_are_exact() -> None:
    # Class 0: 2 GT, 1 hit + 1 predicted as class 1 (FN). Class 1: 1 GT hit + 1 FP.
    metric = _suite(num_classes=2, ignore_index=-1)
    metric.update(
        {
            "seg_pred_labels": torch.tensor([0, 1, 1]),
            "seg_target_labels": torch.tensor([0, 0, 1]),
            "seg_coord": _coord(3),
        }
    )

    test = metric.result(EvalStage.TEST)

    # Class 0: TP=1, FP=0, FN=1 -> IoU 1/2, recall 1/2, precision 1/1, f1 2/3.
    assert test["iou_class_0"] == pytest.approx(0.5)
    assert test["recall_class_0"] == pytest.approx(0.5)
    assert test["precision_class_0"] == pytest.approx(1.0)
    assert test["f1_class_0"] == pytest.approx(2.0 / 3.0)
    # Class 1: TP=1, FP=1, FN=0 -> IoU 1/2, recall 1/1, precision 1/2, f1 2/3.
    assert test["iou_class_1"] == pytest.approx(0.5)
    assert test["recall_class_1"] == pytest.approx(1.0)
    assert test["precision_class_1"] == pytest.approx(0.5)
    assert test["f1_class_1"] == pytest.approx(2.0 / 3.0)
    assert test["acc"] == pytest.approx(2.0 / 3.0)


def test_autoware_seg3d_metric_uses_class_names_when_provided() -> None:
    metric = _suite(num_classes=2, ignore_index=-1, class_names=("car", "road"))
    metric.update(
        {
            "seg_pred_labels": torch.tensor([0, 1]),
            "seg_target_labels": torch.tensor([0, 1]),
            "seg_coord": _coord(2),
        }
    )

    test = metric.result(EvalStage.TEST)

    assert "iou_car" in test and "iou_road" in test
    assert "iou_class_0" not in test


def test_autoware_seg3d_metric_range_buckets_are_exact() -> None:
    ranges = (
        MetricRange("0-50m", 0.0, 50.0),
        MetricRange("50-90m", 50.0, 90.0),
    )
    metric = _suite(num_classes=2, ignore_index=-1, ranges=ranges)
    # Near point (10 m): correct. Far point (60 m): wrong. Both ground-truth class 0.
    metric.update(
        {
            "seg_pred_labels": torch.tensor([0, 1]),
            "seg_target_labels": torch.tensor([0, 0]),
            "seg_coord": torch.tensor([[10.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),
        }
    )

    test = metric.result(EvalStage.TEST)

    # Overall: 1 of 2 correct.
    assert test["acc"] == pytest.approx(0.5)
    # 0-50 m sees only the near (correct) point; 50-90 m only the far (wrong) point.
    assert test["acc_0m_50m"] == pytest.approx(1.0)
    assert test["acc_50m_90m"] == pytest.approx(0.0)
    # Per-class-per-range keys are emitted.
    assert "iou_class_0_0m_50m" in test
    assert test["iou_class_0_0m_50m"] == pytest.approx(1.0)


def test_point_accuracy_skips_ignored_targets() -> None:
    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, -1, 1])

    accuracy = compute_point_accuracy(predictions, targets, ignore_index=-1)

    assert accuracy is not None
    assert torch.isclose(accuracy, torch.tensor(2.0 / 3.0))


def test_point_accuracy_raises_when_all_targets_are_ignored() -> None:
    predictions = torch.tensor([0, 1, 1])
    targets = torch.tensor([-1, -1, -1])

    with pytest.raises(ValueError, match="every target is ignored"):
        compute_point_accuracy(predictions, targets, ignore_index=-1)


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


def test_segmentation_metrics_raises_when_all_ignored() -> None:
    predictions = torch.tensor([0, 1])
    targets = torch.tensor([-1, -1])

    with pytest.raises(ValueError, match="every target is ignored"):
        compute_segmentation_metrics(predictions, targets, num_classes=2, ignore_index=-1)


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
