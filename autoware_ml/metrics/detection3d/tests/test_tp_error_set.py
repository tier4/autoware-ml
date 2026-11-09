"""Tests of the typed true-positive error set, the recall targets and the match cost."""

from __future__ import annotations

import math

import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.matching import (
    Detection3DState,
    cost_center,
    cost_corner,
    cost_function,
    mean_tp_errors,
    nds,
)
from autoware_ml.metrics.detection3d.structures import (
    Detection3DSample,
    RecallTarget,
    TpErrorSet,
)
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.detection3d.tp_errors import TpErrors
from autoware_ml.types.metrics import MatchCost


def _errors(ate: float, aoe: float = 0.0, ase: float = 0.0, ave: float = 0.0) -> TpErrorSet:
    return TpErrorSet(ate=ate, aoe=aoe, ase=ase, ave=ave, aae=1.0)


def test_tp_error_set_names_the_errors_in_nuscenes_order() -> None:
    names = [name for name, _ in _errors(0.5).named()]
    assert names == ["ATE", "AOE", "ASE", "AVE", "AAE"]
    assert dict(_errors(0.5, aoe=0.25).named()) == {
        "ATE": 0.5,
        "AOE": 0.25,
        "ASE": 0.0,
        "AVE": 0.0,
        "AAE": 1.0,
    }


def test_tp_error_set_nan_is_nan_everywhere() -> None:
    assert all(math.isnan(value) for _, value in TpErrorSet.nan().named())


def test_tp_error_set_is_frozen() -> None:
    errors = _errors(0.5)
    with pytest.raises(Exception):
        errors.ate = 1.0  # type: ignore[misc]


def test_mean_tp_errors_averages_each_error_and_skips_nan() -> None:
    mean = mean_tp_errors([_errors(0.2, aoe=1.0), _errors(0.4, aoe=float("nan"))])
    assert mean.ate == pytest.approx(0.3)
    assert mean.aoe == pytest.approx(1.0)
    assert mean.aae == pytest.approx(1.0)
    assert math.isnan(mean_tp_errors([]).ate)


def test_nds_clamps_each_error_at_one() -> None:
    assert nds(1.0, _errors(0.0)) == pytest.approx(0.9)  # aae of 1.0 contributes nothing
    assert nds(1.0, _errors(5.0)) == pytest.approx(0.8)  # an error above 1 scores 0, not negative
    assert nds(0.0, TpErrorSet.nan()) == pytest.approx(0.0)


def test_recall_target_rejects_the_reserved_name_and_an_invalid_recall() -> None:
    assert RecallTarget(name="default", recall=0.1).recall == 0.1
    with pytest.raises(ValueError, match="reserved"):
        RecallTarget(name="optimal", recall=0.1)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        RecallTarget(name="high", recall=1.5)


def test_tp_errors_reports_each_recall_target_and_the_optimal_point() -> None:
    boxes = torch.tensor([[float(i), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0] for i in range(10)])
    sample = Detection3DSample(
        pred_boxes=boxes.clone(),
        pred_scores=torch.tensor([1.0 - 0.01 * i for i in range(10)]),
        pred_labels=torch.zeros(10, dtype=torch.long),
        gt_boxes=boxes.clone(),
        gt_labels=torch.zeros(10, dtype=torch.long),
    )
    state = Detection3DState(samples=[sample], class_names=("car",))
    metric = TpErrors(
        recall_targets=(RecallTarget(name="low", recall=0.1),), thresholds=(2.0,), tp_threshold=2.0
    )
    out = metric.evaluate(state, EvalStage.TEST)
    assert out["mATE_low"] == pytest.approx(0.0)
    assert out["mATE_optimal"] == pytest.approx(0.0)
    # The selector pools the matches up to floor(10 * (0.1 + 0.01)), one match.
    assert out["tp_error_num_match_car_low_2m"] == pytest.approx(1.0)
    assert out["tp_error_num_match_car_optimal_2m"] == pytest.approx(10.0)
    assert "mATE_default" not in out


def test_tp_errors_rejects_duplicate_recall_target_names() -> None:
    with pytest.raises(ValueError, match="distinct"):
        TpErrors(
            recall_targets=(
                RecallTarget(name="low", recall=0.1),
                RecallTarget(name="low", recall=0.4),
            )
        )


def test_cost_function_dispatches_on_the_match_cost() -> None:
    assert cost_function(MatchCost.CENTER) is cost_center
    assert cost_function(MatchCost.CORNER) is cost_corner


def test_suite_accepts_the_match_cost_by_name_and_rejects_a_typo() -> None:
    suite = Detection3DMetricSuite(components=[], class_names=("car",), match_cost="corner")
    assert suite.match_cost is MatchCost.CORNER
    with pytest.raises(ValueError):
        Detection3DMetricSuite(components=[], class_names=("car",), match_cost="centre")
