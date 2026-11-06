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

"""Unit tests for the ground truth boxes the detection suite ignores below its point filter."""

import unittest

import numpy as np
import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.criticality import greedy_match_costs
from autoware_ml.metrics.detection3d.matching import Detection3DState, gt_keep_mask
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.structures import Detection3DSample
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.report import MetricKey, MetricReport
from autoware_ml.metrics.tests.eval_output_fixtures import (
    detection_eval_output,
    detection_frame,
    prediction,
)

_CLASS_NAMES = ("car", "pedestrian")
_GROUPED_NAMES = ("car", "motorcycle", "bicycle")
_GROUPS = {"grouped_vehicle": ["car"], "grouped_two_wheeler": ["motorcycle", "bicycle"]}


def _boxes(*xs: float) -> torch.Tensor:
    """Unit boxes on the x axis."""
    boxes = torch.zeros((len(xs), 9), dtype=torch.float32)
    boxes[:, 0] = torch.tensor(xs)
    boxes[:, 3:6] = 1.0
    return boxes


def _report(
    pred_xs: tuple[float, ...],
    gt_xs: tuple[float, ...],
    gt_points: list[int],
    thresholds: tuple[float, ...] = (2.0,),
) -> MetricReport:
    """Score one frame of cars on the x axis with a point filter of 2."""
    suite = Detection3DMetricSuite(
        components=[BoxCounts(), MeanAP(stages=("val",), thresholds=thresholds)],
        class_names=_CLASS_NAMES,
        min_num_points=2,
        ranges=(MetricRange("0-100m", 0.0, 100.0),),
    )
    suite.update(
        detection_eval_output(
            [
                detection_frame(
                    prediction(
                        _boxes(*pred_xs),
                        torch.linspace(0.9, 0.5, len(pred_xs)),
                        torch.zeros(len(pred_xs), dtype=torch.long),
                    ),
                    _boxes(*gt_xs),
                    torch.zeros(len(gt_xs), dtype=torch.long),
                    torch.tensor(gt_points),
                )
            ]
        )
    )
    return suite.result(EvalStage.VAL)


def _grouped_report(
    gt_xs: tuple[float, ...], gt_labels: list[int], gt_points: list[int], pred_label: int
) -> MetricReport:
    """Score one frame and one prediction at 30.3 m with the two wheelers grouped."""
    suite = Detection3DMetricSuite(
        components=[MeanAP(stages=("val",), thresholds=(2.0,))],
        class_names=_GROUPED_NAMES,
        min_num_points=2,
        class_groups=_GROUPS,
        ranges=(MetricRange("0-100m", 0.0, 100.0),),
    )
    suite.update(
        detection_eval_output(
            [
                detection_frame(
                    prediction(_boxes(30.3), torch.tensor([0.9]), torch.tensor([pred_label])),
                    _boxes(*gt_xs),
                    torch.tensor(gt_labels),
                    torch.tensor(gt_points),
                )
            ]
        )
    )
    return suite.result(EvalStage.VAL)


def _ap(report: MetricReport, name: str = "car") -> float:
    """AP of one class in the report."""
    return report.metric_value(MetricKey(name=f"mAP_{name}"))


class TestIgnoredBoxes(unittest.TestCase):
    """A prediction that claims a box below the point filter counts neither way."""

    def test_the_prediction_on_an_occluded_box_is_no_false_positive(self) -> None:
        """
        Input: a seen car at 10 m, an occluded car at 30 m, predictions on both.
        Expected: the occluded box leaves the ground truth and its prediction the curve.
        Check: one GT box and an AP of 1.
        """
        report = _report((10.5, 30.5), (10.0, 30.0), [5, 1])

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_car")), 1.0)
        self.assertAlmostEqual(_ap(report), 1.0)

    def test_a_prediction_on_empty_road_stays_a_false_positive(self) -> None:
        """
        Input: the frame of the previous test with the second prediction on empty road.
        Expected: a prediction that claims no box stays a false positive and ranks above the
          true positive.
        Check: the AP drops below 1.
        """
        report = _report((50.0, 10.5), (10.0, 30.0), [5, 1])

        self.assertLess(_ap(report), 1.0)

    def test_a_duplicate_on_an_occluded_box_stays_a_false_positive(self) -> None:
        """
        Input: the seen car, two predictions on the occluded car scored above the prediction on
          the seen car.
        Expected: one prediction claims the occluded box, the second is a false positive ranked
          above the true positive.
        Check: the AP drops below 1.
        """
        suite = Detection3DMetricSuite(
            components=[MeanAP(stages=("val",), thresholds=(2.0,))],
            class_names=_CLASS_NAMES,
            min_num_points=2,
            ranges=(MetricRange("0-100m", 0.0, 100.0),),
        )
        suite.update(
            detection_eval_output(
                [
                    detection_frame(
                        prediction(
                            _boxes(30.5, 30.2, 10.5),
                            torch.tensor([0.9, 0.8, 0.7]),
                            torch.zeros(3, dtype=torch.long),
                        ),
                        _boxes(10.0, 30.0),
                        torch.zeros(2, dtype=torch.long),
                        torch.tensor([5, 1]),
                    )
                ]
            )
        )

        self.assertLess(_ap(suite.result(EvalStage.VAL)), 1.0)

    def test_the_claim_follows_the_threshold_of_the_match(self) -> None:
        """
        Input: an occluded car at 30 m and a prediction 1.5 m from it, ahead of the true positive
          on the seen car, matched at 1 m and at 2 m.
        Expected: at 2 m the prediction claims the ignored box, at 1 m it reaches no box and stays a
          false positive.
        Check: the AP is 1 at 2 m and below 1 at 1 m, so the mean lies between.
        """
        report = _report((31.5, 10.5), (10.0, 30.0), [5, 1], thresholds=(1.0, 2.0))

        self.assertLess(_ap(report), 1.0)
        self.assertGreater(_ap(report), _ap(_report((31.5, 10.5), (10.0, 30.0), [5, 1], (1.0,))))

    def test_a_seen_box_takes_its_prediction_before_an_occluded_one(self) -> None:
        """
        Input: a seen car at 10 m, an occluded car at 11 m and one prediction at 10.6 m.
        Expected: the prediction matches the seen box although the occluded box is in reach too.
        Check: an AP of 1.
        """
        self.assertAlmostEqual(_ap(_report((10.6,), (10.0, 11.0), [5, 1])), 1.0)

    def test_without_a_point_filter_nothing_is_ignored(self) -> None:
        """
        Input: a suite without a point filter and a box with one point.
        Expected: the box counts.
        Check: two GT boxes.
        """
        suite = Detection3DMetricSuite(
            components=[BoxCounts()],
            class_names=_CLASS_NAMES,
            ranges=(MetricRange("0-100m", 0.0, 100.0),),
        )
        suite.update(
            detection_eval_output(
                [
                    detection_frame(
                        prediction(_boxes(), torch.zeros(0), torch.zeros(0, dtype=torch.long)),
                        _boxes(10.0, 30.0),
                        torch.zeros(2, dtype=torch.long),
                        torch.tensor([5, 1]),
                    )
                ]
            )
        )

        report = suite.result(EvalStage.VAL)

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_car")), 2.0)


class TestGroupedIgnoredBoxes(unittest.TestCase):
    """With class groups the claims run between the grouped classes, as the matching does."""

    def test_a_prediction_of_a_group_member_claims_an_occluded_box(self) -> None:
        """
        Input: a seen motorcycle at 10 m, an occluded motorcycle at 30 m and a bicycle prediction
          on the occluded one, both two wheelers.
        Expected: the prediction claims the occluded box of its group.
        Check: an AP of 0, the seen box is missed and no false positive ranks.
        """
        report = _grouped_report((10.0, 30.0), [1, 1], [5, 1], 2)

        self.assertAlmostEqual(_ap(report, "grouped_two_wheeler"), 0.0)

    def test_a_prediction_of_another_group_claims_nothing(self) -> None:
        """
        Input: the frame of the previous test with a car prediction on the occluded box.
        Expected: the car prediction is a false positive of its own group.
        Check: the vehicle group has no ground truth and its AP is undefined.
        """
        report = _grouped_report((10.0, 30.0), [1, 1], [5, 1], 0)

        self.assertTrue(np.isnan(_ap(report, "grouped_vehicle")))


class TestGreedyMatchCosts(unittest.TestCase):
    """The claim pass every matching metric shares."""

    def test_a_prediction_without_a_counted_box_claims_an_ignored_one(self) -> None:
        """
        Input: one counted box, one ignored box and three predictions in score order, the first on
          the counted box and the others on the ignored box.
        Expected: the first matches, the second claims the ignored box, the third stays unmatched.
        """
        match = greedy_match_costs(
            np.array([[0.1], [5.0], [5.0]]),
            np.array([[5.0], [0.2], [0.3]]),
            np.array([0.9, 0.8, 0.7]),
            (1.0,),
        )[1.0]

        self.assertEqual(match.is_tp.tolist(), [True, False, False])
        self.assertEqual(match.ignored.tolist(), [False, True, False])
        self.assertEqual(match.matched_gt.tolist(), [0, -1, -1])


class TestIgnoredState(unittest.TestCase):
    """The match curve leaves out the predictions on ignored boxes."""

    def test_the_curve_drops_the_prediction_on_an_ignored_box(self) -> None:
        sample = Detection3DSample(
            pred_boxes=_boxes(10.2, 30.2),
            pred_scores=torch.tensor([0.9, 0.8]),
            pred_labels=torch.zeros(2, dtype=torch.long),
            gt_boxes=_boxes(10.0),
            gt_labels=torch.zeros(1, dtype=torch.long),
            ignored_gt_boxes=_boxes(30.0),
            ignored_gt_labels=torch.zeros(1, dtype=torch.long),
        )
        curve = Detection3DState(samples=[sample], class_names=("car",)).match_curve(0, 1.0)

        self.assertEqual(curve.scores.tolist(), [0.8999999761581421])
        self.assertEqual(curve.true_positive.tolist(), [True])


class TestGtKeepMask(unittest.TestCase):
    """The per-class distance caps and the point filter of the ground truth."""

    def test_caps_only_the_classes_with_a_distance(self) -> None:
        boxes = torch.zeros((5, 9))
        boxes[:, 0] = torch.tensor([40.0, 60.0, 60.0, 60.0, 60.0])
        labels = torch.tensor([0, 0, 1, 2, -1])

        keep = gt_keep_mask(boxes, labels, None, ("car", "truck", "bus"), {"car": 50.0}, 0)

        self.assertEqual(keep.tolist(), [True, False, True, True, True])

    def test_drops_the_boxes_below_the_point_filter(self) -> None:
        boxes = torch.zeros((2, 9))

        keep = gt_keep_mask(boxes, torch.zeros(2), torch.tensor([1, 5]), ("car",), None, 2)

        self.assertEqual(keep.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
