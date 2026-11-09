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

"""Unit tests for the typed true-positive error set, the recall targets and the match cost."""

import math
import unittest

import torch
from pydantic import ValidationError

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
    """An error set with the neutral attribute error."""
    return TpErrorSet(ate=ate, aoe=aoe, ase=ase, ave=ave, aae=1.0)


def _perfect_state() -> Detection3DState:
    """One frame of ten car boxes predicted exactly where they are, scores descending."""
    boxes = torch.tensor([[float(i), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0] for i in range(10)])
    sample = Detection3DSample(
        pred_boxes=boxes.clone(),
        pred_scores=torch.tensor([1.0 - 0.01 * i for i in range(10)]),
        pred_labels=torch.zeros(10, dtype=torch.long),
        gt_boxes=boxes.clone(),
        gt_labels=torch.zeros(10, dtype=torch.long),
    )
    return Detection3DState(samples=[sample], class_names=("car",))


class TestTpErrorSet(unittest.TestCase):
    """The five nuScenes true-positive errors as one typed value."""

    def test_named_lists_the_errors_in_nuscenes_order(self) -> None:
        """
        Input: an error set with a translation error of 0.5 and an orientation error of 0.25.
        Expected: named() pairs every error with its nuScenes name, ATE first and AAE last, so
        the metric keys keep their established order.
        Check: the names and the paired values.
        """
        named = _errors(0.5, aoe=0.25).named()
        self.assertEqual([name for name, _ in named], ["ATE", "AOE", "ASE", "AVE", "AAE"])
        self.assertEqual(dict(named), {"ATE": 0.5, "AOE": 0.25, "ASE": 0.0, "AVE": 0.0, "AAE": 1.0})

    def test_nan_set_is_nan_in_every_error(self) -> None:
        """
        Input: the error set of a selection without a measurable match.
        Expected: every error is NaN, so an empty bucket reports no number instead of a fake one.
        Check: each named value is NaN.
        """
        for name, value in TpErrorSet.nan().named():
            with self.subTest(error=name):
                self.assertTrue(math.isnan(value))

    def test_set_is_frozen(self) -> None:
        """
        Input: a built error set.
        Expected: an error cannot be reassigned after construction.
        Check: pydantic raises on the assignment.
        """
        errors = _errors(0.5)
        with self.assertRaises(ValidationError):
            errors.ate = 1.0  # type: ignore[misc]


class TestMeanTpErrorsAndNds(unittest.TestCase):
    """Averaging error sets across classes and folding them into the detection score."""

    def test_mean_averages_each_error_and_skips_nan(self) -> None:
        """
        Input: two error sets, translation errors 0.2 and 0.4, the second orientation error NaN.
        Expected: each error is averaged on its own and a NaN entry leaves the mean of the others.
        Check: the mean translation, orientation and attribute errors.
        """
        mean = mean_tp_errors([_errors(0.2, aoe=1.0), _errors(0.4, aoe=float("nan"))])
        self.assertAlmostEqual(mean.ate, 0.3)
        self.assertAlmostEqual(mean.aoe, 1.0)
        self.assertAlmostEqual(mean.aae, 1.0)

    def test_mean_of_no_sets_is_nan(self) -> None:
        """
        Input: no error sets.
        Expected: the mean has no basis and is NaN.
        Check: the mean translation error.
        """
        self.assertTrue(math.isnan(mean_tp_errors([]).ate))

    def test_nds_clamps_each_error_at_one(self) -> None:
        """
        Input: a perfect mean AP with zero errors, with a translation error of 5.0, and a zero
        mean AP with NaN errors.
        Expected: the neutral attribute error of 1.0 contributes nothing, an error above 1.0
        scores 0 instead of going negative, and NaN errors contribute nothing.
        Check: the three detection scores.
        """
        self.assertAlmostEqual(nds(1.0, _errors(0.0)), 0.9)
        self.assertAlmostEqual(nds(1.0, _errors(5.0)), 0.8)
        self.assertAlmostEqual(nds(0.0, TpErrorSet.nan()), 0.0)


class TestRecallTarget(unittest.TestCase):
    """A recall operating point validates itself on construction."""

    def test_accepts_a_named_recall_in_the_unit_range(self) -> None:
        """
        Input: the default recall target at 0.1.
        Expected: it is stored as given.
        Check: the name and the recall.
        """
        target = RecallTarget(name="default", recall=0.1)
        self.assertEqual(target.name, "default")
        self.assertEqual(target.recall, 0.1)

    def test_rejects_the_reserved_name(self) -> None:
        """
        Input: a recall target named optimal.
        Expected: it is refused, that name is the max-F1 operating point TpErrors always reports.
        Check: ValueError naming the reservation.
        """
        with self.assertRaisesRegex(ValueError, "reserved"):
            RecallTarget(name="optimal", recall=0.1)

    def test_rejects_a_recall_outside_the_unit_range(self) -> None:
        """
        Input: a recall target at 1.5.
        Expected: it is refused, a recall is a fraction of the ground truth.
        Check: ValueError naming the range.
        """
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            RecallTarget(name="high", recall=1.5)


class TestTpErrorsMetric(unittest.TestCase):
    """The TP error metric reports the configured recall targets and the optimal point."""

    def test_reports_each_recall_target_and_the_optimal_point(self) -> None:
        """
        Input: ten perfectly predicted car boxes and one recall target named low at 0.1.
        Expected: the aggregate errors and the per-class match counts appear for the low variant
        and the optimal variant, and nothing for the default variant that was not configured.
        The low variant pools floor(10 * (0.1 + 0.01)) matches, one, the optimal point all ten.
        Check: the mean translation errors, the match counts and the absent default key.
        """
        metric = TpErrors(
            recall_targets=(RecallTarget(name="low", recall=0.1),),
            thresholds=(2.0,),
            tp_threshold=2.0,
        )
        out = metric.evaluate(_perfect_state(), EvalStage.TEST)
        self.assertAlmostEqual(out["mATE_low"], 0.0)
        self.assertAlmostEqual(out["mATE_optimal"], 0.0)
        self.assertAlmostEqual(out["tp_error_num_match_car_low_2m"], 1.0)
        self.assertAlmostEqual(out["tp_error_num_match_car_optimal_2m"], 10.0)
        self.assertNotIn("mATE_default", out)

    def test_rejects_duplicate_recall_target_names(self) -> None:
        """
        Input: two recall targets both named low.
        Expected: the metric refuses them, their keys would collide.
        Check: ValueError naming the distinctness rule.
        """
        with self.assertRaisesRegex(ValueError, "distinct"):
            TpErrors(
                recall_targets=(
                    RecallTarget(name="low", recall=0.1),
                    RecallTarget(name="low", recall=0.4),
                )
            )


class TestMatchCost(unittest.TestCase):
    """The match cost is an enum the suite parses from its config name."""

    def test_cost_function_dispatches_on_the_match_cost(self) -> None:
        """
        Input: the two match costs.
        Expected: each resolves to its cost function.
        Check: identity of the returned functions.
        """
        self.assertIs(cost_function(MatchCost.CENTER), cost_center)
        self.assertIs(cost_function(MatchCost.CORNER), cost_corner)

    def test_suite_accepts_the_match_cost_by_name(self) -> None:
        """
        Input: a detection suite configured with the name corner, as a YAML config spells it.
        Expected: the suite stores the enum member.
        Check: identity of the stored match cost.
        """
        suite = Detection3DMetricSuite(components=[], class_names=("car",), match_cost="corner")
        self.assertIs(suite.match_cost, MatchCost.CORNER)

    def test_suite_rejects_a_misspelled_match_cost(self) -> None:
        """
        Input: a detection suite configured with the misspelled name centre.
        Expected: the suite refuses it at build, not after the epoch.
        Check: ValueError.
        """
        with self.assertRaises(ValueError):
            Detection3DMetricSuite(components=[], class_names=("car",), match_cost="centre")


if __name__ == "__main__":
    unittest.main()
