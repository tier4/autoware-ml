"""Tests for 3D semantic segmentation metrics."""

from __future__ import annotations

import torch
import pytest

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.frame_count import FrameCount
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.precision_recall_f1 import PrecisionRecallF1
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite
from autoware_ml.metrics.report import MetricKey


def _suite(**kwargs):
    """Build a segmentation suite with the full standard metric set injected."""
    return Segmentation3DConfusionMatrixMetricSuite(
        components=[IoU(), Accuracy(), PrecisionRecallF1()],
        **kwargs,
    )


def _eval_out(pred, target, coord):
    """One-frame seg3d_frames payload (the contract both seg suites consume)."""
    return {"seg3d_frames": [{"pred": pred, "target": target, "coord": coord}]}


def _coord(num_points: int) -> torch.Tensor:
    """Origin-centered point coordinates for tests where range does not matter."""
    return torch.zeros(num_points, 3)


def test_autoware_seg3d_metric_validation_reports_headline_scalars() -> None:
    metric = _suite(num_classes=3, ignore_index=-1)
    metric.update(
        _eval_out(torch.tensor([0, 0, 1, 1, 2, -1]), torch.tensor([0, 0, 1, 1, 2, -1]), _coord(6))
    )

    val = metric.result(EvalStage.VAL)

    # Validation reports exactly the three headline scalars, nothing per class.
    assert set(val.metric_names()) == {
        MetricKey(name="mIoU"),
        MetricKey(name="acc"),
        MetricKey(name="mRecall"),
    }
    assert val.value(MetricKey(name="mIoU")) == pytest.approx(1.0)
    assert val.value(MetricKey(name="acc")) == pytest.approx(1.0)
    assert val.value(MetricKey(name="mRecall")) == pytest.approx(1.0)


def test_autoware_seg3d_metric_test_reports_full_and_per_class() -> None:
    metric = _suite(num_classes=3, ignore_index=-1)
    metric.update(
        _eval_out(torch.tensor([0, 0, 1, 1, 2, -1]), torch.tensor([0, 0, 1, 1, 2, -1]), _coord(6))
    )

    test = metric.result(EvalStage.TEST)

    for scalar in ("mIoU", "acc", "mRecall", "fwIoU", "mPrecision", "mF1"):
        assert test.value(MetricKey(name=scalar)) == pytest.approx(1.0)
    for class_index in range(3):
        for family in ("iou", "recall", "precision", "f1"):
            assert test.value(MetricKey(name=f"{family}_class_{class_index}")) == pytest.approx(1.0)


def test_autoware_seg3d_metric_excludes_classes_without_support() -> None:
    # Class 2 never appears in the ground truth, so it must not be reported.
    metric = _suite(num_classes=3, ignore_index=-1)
    metric.update(_eval_out(torch.tensor([0, 1]), torch.tensor([0, 1]), _coord(2)))

    test = metric.result(EvalStage.TEST)

    assert MetricKey(name="iou_class_2") not in test.metric_names()
    assert (
        MetricKey(name="iou_class_0") in test.metric_names()
        and MetricKey(name="iou_class_1") in test.metric_names()
    )


def test_autoware_seg3d_metric_per_class_values_are_exact() -> None:
    # Class 0: 2 GT, 1 hit + 1 predicted as class 1 (FN). Class 1: 1 GT hit + 1 FP.
    metric = _suite(num_classes=2, ignore_index=-1)
    metric.update(_eval_out(torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]), _coord(3)))

    test = metric.result(EvalStage.TEST)

    # Class 0: TP=1, FP=0, FN=1 -> IoU 1/2, recall 1/2, precision 1/1, f1 2/3.
    assert test.value(MetricKey(name="iou_class_0")) == pytest.approx(0.5)
    assert test.value(MetricKey(name="recall_class_0")) == pytest.approx(0.5)
    assert test.value(MetricKey(name="precision_class_0")) == pytest.approx(1.0)
    assert test.value(MetricKey(name="f1_class_0")) == pytest.approx(2.0 / 3.0)
    # Class 1: TP=1, FP=1, FN=0 -> IoU 1/2, recall 1/1, precision 1/2, f1 2/3.
    assert test.value(MetricKey(name="iou_class_1")) == pytest.approx(0.5)
    assert test.value(MetricKey(name="recall_class_1")) == pytest.approx(1.0)
    assert test.value(MetricKey(name="precision_class_1")) == pytest.approx(0.5)
    assert test.value(MetricKey(name="f1_class_1")) == pytest.approx(2.0 / 3.0)
    assert test.value(MetricKey(name="acc")) == pytest.approx(2.0 / 3.0)


def test_autoware_seg3d_metric_uses_class_names_when_provided() -> None:
    metric = _suite(num_classes=2, ignore_index=-1, class_names=("car", "road"))
    metric.update(_eval_out(torch.tensor([0, 1]), torch.tensor([0, 1]), _coord(2)))

    test = metric.result(EvalStage.TEST)

    assert (
        MetricKey(name="iou_car") in test.metric_names()
        and MetricKey(name="iou_road") in test.metric_names()
    )
    assert MetricKey(name="iou_class_0") not in test.metric_names()


def test_autoware_seg3d_metric_range_buckets_are_exact() -> None:
    ranges = (
        MetricRange("0-50m", 0.0, 50.0),
        MetricRange("50-90m", 50.0, 90.0),
    )
    metric = _suite(num_classes=2, ignore_index=-1, ranges=ranges)
    # Near point (10 m): correct. Far point (60 m): wrong. Both ground-truth class 0.
    metric.update(
        _eval_out(
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[10.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),
        )
    )

    test = metric.result(EvalStage.TEST)

    # Overall: 1 of 2 correct.
    assert test.value(MetricKey(name="acc")) == pytest.approx(0.5)
    # 0-50 m sees only the near (correct) point; 50-90 m only the far (wrong) point.
    assert test.value(MetricKey(name="acc", range_suffix="0m_50m")) == pytest.approx(1.0)
    assert test.value(MetricKey(name="acc", range_suffix="50m_90m")) == pytest.approx(0.0)
    # Per-class-per-range keys are emitted.
    assert MetricKey(name="iou_class_0", range_suffix="0m_50m") in test.metric_names()
    assert test.value(MetricKey(name="iou_class_0", range_suffix="0m_50m")) == pytest.approx(1.0)


def test_autoware_seg3d_frame_count_sums_frames_across_batches() -> None:
    metric = Segmentation3DConfusionMatrixMetricSuite(
        components=[FrameCount()],
        num_classes=2,
        ignore_index=-1,
        ranges=(MetricRange("0-50m", 0.0, 50.0),),
    )
    metric.update(
        {
            "seg3d_frames": [
                {"pred": torch.tensor([0, 1]), "target": torch.tensor([0, 1]), "coord": _coord(2)},
                {"pred": torch.tensor([1]), "target": torch.tensor([0]), "coord": _coord(1)},
            ]
        }
    )
    # A frame whose points are all ignored adds nothing to the matrix but is still a frame.
    metric.update(_eval_out(torch.tensor([0]), torch.tensor([-1]), _coord(1)))

    val = metric.result(EvalStage.VAL)

    assert val.value(MetricKey(name="num_frames")) == 3.0
    # Range windows clip points, never frames, so the count is reported once, not per window.
    assert MetricKey(name="num_frames", range_suffix="0m_50m") not in val.metric_names()


def test_autoware_seg3d_frame_count_resets_with_the_suite() -> None:
    metric = Segmentation3DConfusionMatrixMetricSuite(
        components=[FrameCount()], num_classes=2, ignore_index=-1, ranges=()
    )
    metric.update(_eval_out(torch.tensor([0]), torch.tensor([0]), _coord(1)))
    metric.update(_eval_out(torch.tensor([0]), torch.tensor([0]), _coord(1)))
    metric.reset()
    metric.update(_eval_out(torch.tensor([0]), torch.tensor([0]), _coord(1)))

    assert metric.result(EvalStage.TEST).value(MetricKey(name="num_frames")) == 1.0
