"""Unit tests for the single-responsibility metrics.

Each metric is exercised directly against a hand-built state (a ``DetectionState``
or a ``ConfusionState``), independent of the suites. This pins the stage gating
and the exact key slice each metric owns. A final test checks the suite-level
stage filtering driven by each metric's ``stages``.
"""

from __future__ import annotations

import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.heading_ap import HeadingAP
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.nds import Nds
from autoware_ml.metrics.detection3d.structures import Detection3DSample, DetectionState
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.detection3d.tp_errors import TpErrors
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.precision_recall_f1 import PrecisionRecallF1


def _det_state(class_names=("car",), thresholds=(0.5,)) -> DetectionState:
    box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    sample = Detection3DSample(
        pred_boxes=box.clone(),
        pred_scores=torch.tensor([0.9]),
        pred_labels=torch.tensor([0]),
        gt_boxes=box.clone(),
        gt_labels=torch.tensor([0]),
    )
    return DetectionState(samples=[sample], class_names=class_names, thresholds=thresholds)


def test_mean_ap_validation_emits_only_map_keys() -> None:
    out = MeanAP().evaluate(_det_state(), EvalStage.VAL)
    assert set(out) == {"mAP", "mAP_car"}
    assert out["mAP"] == pytest.approx(1.0)


def test_mean_ap_test_adds_curve_details() -> None:
    out = MeanAP().evaluate(_det_state(), EvalStage.TEST)
    assert {"mAP", "mAP_car", "gt_count_car", "AP_car_0p5m", "num_match_car_0p5m"} <= set(out)
    assert out["num_match_car_0p5m"] == pytest.approx(1.0)


def test_heading_ap_is_test_only() -> None:
    assert HeadingAP().stages == frozenset({EvalStage.TEST})
    out = HeadingAP().evaluate(_det_state(), EvalStage.TEST)
    assert out["mAPH"] == pytest.approx(1.0)


def test_nds_is_test_only_and_combines_ap_with_errors() -> None:
    assert Nds().stages == frozenset({EvalStage.TEST})
    out = Nds().evaluate(_det_state(), EvalStage.TEST)
    assert set(out) == {"map_based_nds", "mapH_based_nds"}


def test_tp_errors_emits_mean_and_per_class() -> None:
    out = TpErrors().evaluate(_det_state(), EvalStage.TEST)
    assert "mATE_default" in out
    assert "ATE_car_default_0p5m" in out


def _conf_state(num_classes: int = 2) -> ConfusionState:
    # Class 0: TP=1, FN=1; class 1: TP=1, FP=1.
    confusion = torch.tensor([[1, 1], [0, 1]], dtype=torch.long)
    return ConfusionState(confusion=confusion, class_names=("car", "road"), num_classes=num_classes)


def test_iou_validation_emits_only_miou() -> None:
    out = IoU().evaluate(_conf_state(), EvalStage.VAL)
    assert set(out) == {"mIoU"}


def test_iou_test_adds_fwiou_and_per_class() -> None:
    out = IoU().evaluate(_conf_state(), EvalStage.TEST)
    assert {"mIoU", "fwIoU", "iou_car", "iou_road"} <= set(out)
    assert out["iou_car"] == pytest.approx(0.5)


def test_accuracy_is_a_single_global_ratio() -> None:
    out = Accuracy().evaluate(_conf_state(), EvalStage.TEST)
    assert set(out) == {"acc"}
    assert out["acc"] == pytest.approx(2.0 / 3.0)


def test_precision_recall_f1_validation_emits_only_mrecall() -> None:
    out = PrecisionRecallF1().evaluate(_conf_state(), EvalStage.VAL)
    assert set(out) == {"mRecall"}


def test_precision_recall_f1_test_adds_macro_and_per_class() -> None:
    out = PrecisionRecallF1().evaluate(_conf_state(), EvalStage.TEST)
    assert {"mRecall", "mPrecision", "mF1", "recall_car", "precision_car", "f1_car"} <= set(out)


def test_suite_gates_metrics_by_configured_stages() -> None:
    """A metric whose stages exclude val must not emit anything at val."""
    box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    suite = Detection3DMetricSuite(
        components=[MeanAP(stages=["val", "test"]), TpErrors(stages=["test"])],
        thresholds=(0.5,),
        class_names=("car",),
        ranges=(),
    )
    suite.update(
        {
            "predictions": [
                {
                    "bboxes_3d": box.clone(),
                    "scores_3d": torch.tensor([0.9]),
                    "labels_3d": torch.tensor([0]),
                }
            ],
            "gt_boxes": [box.clone()],
            "gt_labels": [torch.tensor([0])],
        }
    )

    val = suite.result(EvalStage.VAL)
    assert "mAP" in val
    assert not any(key.startswith("mATE") for key in val)

    suite.reset()
    suite.update(
        {
            "predictions": [
                {
                    "bboxes_3d": box.clone(),
                    "scores_3d": torch.tensor([0.9]),
                    "labels_3d": torch.tensor([0]),
                }
            ],
            "gt_boxes": [box.clone()],
            "gt_labels": [torch.tensor([0])],
        }
    )
    test = suite.result(EvalStage.TEST)
    assert "mAP" in test
    assert "mATE_default" in test
