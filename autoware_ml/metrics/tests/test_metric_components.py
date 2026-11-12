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

"""Unit tests for the single-responsibility metrics.

Each metric is exercised directly against a hand-built state (a ``Detection3DState``
or a ``ConfusionState``), independent of the suites. This pins the stage gating
and the exact key slice each metric owns. A final test checks the suite-level
stage filtering driven by each metric's ``stages``.
"""

from __future__ import annotations

import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.calibration import CalibrationError
from autoware_ml.metrics.detection3d.confident_error import ConfidentErrorRate
from autoware_ml.metrics.detection3d.heading_ap import HeadingAP
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.nds import Nds
from autoware_ml.metrics.detection3d.matching import Detection3DState
from autoware_ml.metrics.detection3d.structures import Detection3DSample
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.detection3d.tp_errors import TpErrors
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.precision_recall_f1 import PrecisionRecallF1
from autoware_ml.metrics.report import MetricKey
from autoware_ml.metrics.tests.eval_output_fixtures import (
    detection_eval_output,
    detection_frame,
    prediction,
)


def _det3d_state(class_names=("car",)) -> Detection3DState:
    box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    sample = Detection3DSample(
        pred_boxes=box.clone(),
        pred_scores=torch.tensor([0.9]),
        pred_labels=torch.tensor([0]),
        gt_boxes=box.clone(),
        gt_labels=torch.tensor([0]),
    )
    return Detection3DState(samples=[sample], class_names=class_names)


def test_mean_ap_validation_emits_only_map_keys() -> None:
    out = MeanAP().evaluate(_det3d_state(), EvalStage.VAL)
    assert set(out.metric_names()) == {MetricKey(name="mAP"), MetricKey(name="mAP_car")}
    assert out.metric_value(MetricKey(name="mAP")) == pytest.approx(1.0)


def test_mean_ap_test_adds_curve_details() -> None:
    out = MeanAP().evaluate(_det3d_state(), EvalStage.TEST)
    assert {
        MetricKey(name="mAP"),
        MetricKey(name="mAP_car"),
        MetricKey(name="gt_count_car"),
        MetricKey(name="AP_car_0p5m"),
        MetricKey(name="num_match_car_0p5m"),
    } <= set(out.metric_names())
    assert out.metric_value(MetricKey(name="num_match_car_0p5m")) == pytest.approx(1.0)


def test_heading_ap_is_test_only() -> None:
    assert HeadingAP().stages == frozenset({EvalStage.TEST})
    out = HeadingAP().evaluate(_det3d_state(), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="mAPH")) == pytest.approx(1.0)


def test_nds_is_test_only_and_combines_ap_with_errors() -> None:
    assert Nds().stages == frozenset({EvalStage.TEST})
    out = Nds().evaluate(_det3d_state(), EvalStage.TEST)
    assert set(out.metric_names()) == {
        MetricKey(name="map_based_nds"),
        MetricKey(name="mapH_based_nds"),
    }


def test_tp_errors_emits_mean_and_per_class() -> None:
    out = TpErrors().evaluate(_det3d_state(), EvalStage.TEST)
    assert MetricKey(name="mATE_default") in out.metric_names()
    assert MetricKey(name="ATE_car_default_0p5m") in out.metric_names()


def test_tp_error_means_ignore_classes_without_true_positives() -> None:
    # car has 10 perfect matches (selected at the recall target); truck is absent.
    # The absent class must not drag the mean TP errors / NDS to the worst case.
    boxes = torch.tensor([[float(i), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0] for i in range(10)])
    sample = Detection3DSample(
        pred_boxes=boxes.clone(),
        pred_scores=torch.tensor([1.0 - 0.01 * i for i in range(10)]),
        pred_labels=torch.zeros(10, dtype=torch.long),
        gt_boxes=boxes.clone(),
        gt_labels=torch.zeros(10, dtype=torch.long),
    )
    state = Detection3DState(samples=[sample], class_names=("car", "truck"))

    nds_out = Nds().evaluate(state, EvalStage.TEST)
    assert nds_out.metric_value(MetricKey(name="map_based_nds")) == pytest.approx(
        0.9
    )  # 0.7 if the absent truck leaked in

    tp_out = TpErrors().evaluate(state, EvalStage.TEST)
    assert tp_out.metric_value(MetricKey(name="mATE_default")) == pytest.approx(0.0)
    assert tp_out.metric_value(MetricKey(name="mAAE_default")) == pytest.approx(1.0)


def _det3d_state_tp_fp() -> Detection3DState:
    # One prediction on the GT (true positive, score 0.9) and one far away with no
    # GT (false positive, score 0.8).
    gt = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    preds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0],
        ]
    )
    sample = Detection3DSample(
        pred_boxes=preds,
        pred_scores=torch.tensor([0.9, 0.8]),
        pred_labels=torch.tensor([0, 0]),
        gt_boxes=gt,
        gt_labels=torch.tensor([0]),
    )
    return Detection3DState(samples=[sample], class_names=("car",))


def test_det3d_confident_error_flags_high_score_false_positive() -> None:
    out = ConfidentErrorRate(score_threshold=0.5).evaluate(_det3d_state_tp_fp(), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="confident_error_rate")) == pytest.approx(
        1.0
    )  # the one FP (0.8) is confident
    assert out.metric_value(MetricKey(name="confident_error_count")) == pytest.approx(1.0)
    lifted = ConfidentErrorRate(score_threshold=0.9).evaluate(_det3d_state_tp_fp(), EvalStage.TEST)
    assert lifted.metric_value(MetricKey(name="confident_error_rate")) == pytest.approx(
        0.0
    )  # 0.8 < 0.9


def test_det3d_calibration_matches_hand_computed_ece() -> None:
    # TP score 0.9 (correct) -> bin 13, gap |1-0.9|=0.1. FP score 0.8 (wrong) -> bin 12,
    # gap |0-0.8|=0.8. Each bin weight 1/2 -> ECE = 0.5*0.1 + 0.5*0.8 = 0.45.
    out = CalibrationError(num_bins=15).evaluate(_det3d_state_tp_fp(), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="ece")) == pytest.approx(0.45)
    assert MetricKey(name="ece_macro") in out.metric_names()


def test_suite_prefix_override_for_side_by_side_suites() -> None:
    # B3 runs a second detection suite alongside the main one, a per-instance
    # prefix keeps their keys in distinct namespaces (the mixin fails on a clash).
    default = Detection3DMetricSuite(
        components=[MeanAP(stages=["test"])], class_names=("car",), ranges=()
    )
    visible = Detection3DMetricSuite(
        components=[MeanAP(stages=["test"])],
        class_names=("car",),
        ranges=(),
        prefix="det3d_visible",
        min_num_points=1,
    )
    assert default.prefix == "det3d"
    assert visible.prefix == "det3d_visible"
    assert visible.min_num_points == 1


def _conf_state(num_classes: int = 2) -> ConfusionState:
    # Class 0: TP=1, FN=1; class 1: TP=1, FP=1.
    confusion = torch.tensor([[1, 1], [0, 1]], dtype=torch.long)
    return ConfusionState(
        confusion=confusion, class_names=("car", "road"), num_classes=num_classes, num_frames=1
    )


def test_iou_validation_emits_only_miou() -> None:
    out = IoU().evaluate(_conf_state(), EvalStage.VAL)
    assert set(out.metric_names()) == {MetricKey(name="mIoU")}


def test_iou_test_adds_fwiou_and_per_class() -> None:
    out = IoU().evaluate(_conf_state(), EvalStage.TEST)
    assert {
        MetricKey(name="mIoU"),
        MetricKey(name="fwIoU"),
        MetricKey(name="iou_car"),
        MetricKey(name="iou_road"),
    } <= set(out.metric_names())
    assert out.metric_value(MetricKey(name="iou_car")) == pytest.approx(0.5)


def test_accuracy_is_a_single_global_ratio() -> None:
    out = Accuracy().evaluate(_conf_state(), EvalStage.TEST)
    assert set(out.metric_names()) == {MetricKey(name="acc")}
    assert out.metric_value(MetricKey(name="acc")) == pytest.approx(2.0 / 3.0)


def test_precision_recall_f1_validation_emits_only_mrecall() -> None:
    out = PrecisionRecallF1().evaluate(_conf_state(), EvalStage.VAL)
    assert set(out.metric_names()) == {MetricKey(name="mRecall")}


def test_precision_recall_f1_test_adds_macro_and_per_class() -> None:
    out = PrecisionRecallF1().evaluate(_conf_state(), EvalStage.TEST)
    assert {
        MetricKey(name="mRecall"),
        MetricKey(name="mPrecision"),
        MetricKey(name="mF1"),
        MetricKey(name="recall_car"),
        MetricKey(name="precision_car"),
        MetricKey(name="f1_car"),
    } <= set(out.metric_names())


def test_suite_gates_metrics_by_configured_stages() -> None:
    """A metric whose stages exclude val must not emit anything at val."""
    box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    suite = Detection3DMetricSuite(
        components=[
            MeanAP(stages=["val", "test"], thresholds=(0.5,)),
            TpErrors(stages=["test"], thresholds=(0.5,), tp_threshold=0.5),
        ],
        class_names=("car",),
        ranges=(),
    )
    one_matched_box = detection_eval_output(
        [
            detection_frame(
                predictions=prediction(box.clone(), torch.tensor([0.9]), torch.tensor([0])),
                gt_boxes=box.clone(),
                gt_labels=torch.tensor([0]),
            )
        ]
    )
    suite.update(one_matched_box)

    val = suite.result(EvalStage.VAL)
    assert MetricKey(name="mAP") in val.metric_names()
    assert not any(key.name.startswith("mATE") for key in val.metric_names())

    suite.reset()
    suite.update(one_matched_box)
    test = suite.result(EvalStage.TEST)
    assert MetricKey(name="mAP") in test.metric_names()
    assert MetricKey(name="mATE_default") in test.metric_names()
