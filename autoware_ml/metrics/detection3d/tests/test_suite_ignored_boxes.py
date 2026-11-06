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
from math import pi

import torch

from autoware_ml.metrics.base import EvalStage, Metric, MetricRange
from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.matching import (
    Detection3DState,
    cost_center,
    cost_corner,
    gt_keep_mask,
    ignored_prediction_mask,
)
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


def _long_box(x: float, yaw: float) -> torch.Tensor:
    """One 4 m by 1 m box on the x axis."""
    box = torch.zeros((1, 9), dtype=torch.float32)
    box[0, 0] = x
    box[0, 3:6] = torch.tensor([4.0, 1.0, 1.5])
    box[0, 6] = yaw
    return box


def _score(
    suite: Detection3DMetricSuite,
    pred_xs: tuple[float, ...],
    gt_xs: tuple[float, ...],
    gt_points: list[int],
) -> MetricReport:
    """Score one frame of cars on the x axis."""
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


def _report(ignore_below_min_points: bool, pred_xs: tuple[float, ...]) -> dict[str, float]:
    """Count one frame with a seen car at 10 m and an occluded car at 30 m."""
    suite = Detection3DMetricSuite(
        components=[BoxCounts()],
        class_names=_CLASS_NAMES,
        min_num_points=2,
        ignore_below_min_points=ignore_below_min_points,
        ranges=(MetricRange("0-100m", 0.0, 100.0),),
    )
    return _score(suite, pred_xs, (10.0, 30.0), [5, 1])


def _grouped_report(
    gt_xs: tuple[float, ...],
    gt_labels: list[int],
    gt_points: list[int],
    pred_x: float,
    pred_label: int,
) -> MetricReport:
    """Score one frame and one prediction with the two wheelers grouped."""
    suite = Detection3DMetricSuite(
        components=[BoxCounts(), MeanAP(stages=("val",))],
        class_names=_GROUPED_NAMES,
        min_num_points=2,
        ignore_below_min_points=True,
        class_groups=_GROUPS,
        ranges=(MetricRange("0-100m", 0.0, 100.0),),
    )
    suite.update(
        detection_eval_output(
            [
                detection_frame(
                    prediction(_boxes(pred_x), torch.tensor([0.9]), torch.tensor([pred_label])),
                    _boxes(*gt_xs),
                    torch.tensor(gt_labels),
                    torch.tensor(gt_points),
                )
            ]
        )
    )
    return suite.result(EvalStage.VAL)


class TestIgnoredBoxes(unittest.TestCase):
    """A prediction on a box below the point filter counts neither way when ignored."""

    def test_without_the_policy_the_prediction_on_an_occluded_box_stays(self) -> None:
        """
        Input: predictions on the seen car, on the occluded car and on empty road.
        Expected: the occluded box is dropped and its prediction kept as a false positive.
        Check: one GT box and three predictions.
        """
        report = _report(False, (10.5, 30.5, 50.0))

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_car")), 1.0)
        self.assertEqual(report.metric_value(MetricKey(name="num_preds_car")), 3.0)

    def test_the_policy_leaves_out_the_prediction_on_an_occluded_box(self) -> None:
        """
        Input: predictions on the seen car, on the occluded car and on empty road.
        Expected: the prediction on the occluded box leaves with it, the one on empty road stays.
        Check: one GT box and two predictions.
        """
        report = _report(True, (10.5, 30.5, 50.0))

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_car")), 1.0)
        self.assertEqual(report.metric_value(MetricKey(name="num_preds_car")), 2.0)

    def test_a_duplicate_on_an_occluded_box_stays_a_false_positive(self) -> None:
        """
        Input: two predictions on the occluded car.
        Expected: only the higher scored one claims the box and leaves.
        Check: one prediction left.
        """
        report = _report(True, (30.5, 30.2))

        self.assertEqual(report.metric_value(MetricKey(name="num_preds_car")), 1.0)

    def test_a_seen_box_keeps_its_prediction_up_to_the_largest_match_threshold(self) -> None:
        """
        Input: a seen car at 10 m, an occluded car at 14 m and a prediction at 13 m, scored by
        MeanAP at 0.5, 1, 2 and 4 m.
        Expected: the seen box takes the prediction within 4 m, so it matches at 4 m.
        Check: the kept reach of the suite and an AP of 1 at one of the four thresholds.
        """
        suite = Detection3DMetricSuite(
            components=[MeanAP(stages=("val",))],
            class_names=_CLASS_NAMES,
            min_num_points=2,
            ignore_below_min_points=True,
            ranges=(MetricRange("0-100m", 0.0, 100.0),),
        )

        report = _score(suite, (13.0,), (10.0, 14.0), [5, 1])

        self.assertEqual(suite.kept_claim_distance, 4.0)
        self.assertAlmostEqual(report.metric_value(MetricKey(name="mAP_car")), 0.25)

    def test_every_component_declares_its_match_distance(self) -> None:
        """
        Input: a suite with a detection metric that is not a Detection3DMetric.
        Expected: the suite refuses to build and names the component.
        """

        class _Plain(Metric[Detection3DState]):
            def evaluate(self, state: Detection3DState, stage: EvalStage) -> MetricReport:
                return MetricReport()

        with self.assertRaisesRegex(TypeError, "_Plain"):
            Detection3DMetricSuite(components=[_Plain()], class_names=_CLASS_NAMES)

    def test_the_policy_needs_a_point_filter(self) -> None:
        """
        Input: the policy without a min_num_points filter.
        Expected: the suite refuses to build, there is nothing to ignore.
        Check: ValueError.
        """
        with self.assertRaisesRegex(ValueError, "min_num_points"):
            Detection3DMetricSuite(
                components=[BoxCounts()], class_names=_CLASS_NAMES, ignore_below_min_points=True
            )


class TestGroupedIgnoredBoxes(unittest.TestCase):
    """With class groups the claims run between the grouped classes, as the matching does."""

    def test_a_prediction_of_a_group_member_claims_an_occluded_box(self) -> None:
        """
        Input: an occluded motorcycle and a bicycle prediction on it, both two wheelers.
        Expected: the prediction claims the occluded box and leaves with it.
        Check: no GT box and no prediction of the group.
        """
        report = _grouped_report((30.0,), [1], [1], 30.3, 2)

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_grouped_two_wheeler")), 0.0)
        self.assertEqual(report.metric_value(MetricKey(name="num_preds_grouped_two_wheeler")), 0.0)

    def test_a_group_member_prediction_claims_the_nearer_seen_box(self) -> None:
        """
        Input: a seen motorcycle at 10 m, an occluded bicycle at 11 m, a bicycle prediction at
        10.4 m.
        Expected: the prediction claims the nearer seen box and matches it.
        Check: one GT box, one prediction and an AP of 1.
        """
        report = _grouped_report((10.0, 11.0), [1, 2], [5, 1], 10.4, 2)

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_grouped_two_wheeler")), 1.0)
        self.assertEqual(report.metric_value(MetricKey(name="num_preds_grouped_two_wheeler")), 1.0)
        self.assertAlmostEqual(report.metric_value(MetricKey(name="mAP_grouped_two_wheeler")), 1.0)


class TestIgnoredPredictionMask(unittest.TestCase):
    """Kept boxes take their predictions first, the predictions left over claim ignored boxes."""

    def test_the_claims_follow_the_match_cost_of_the_suite(self) -> None:
        """
        Input: a seen car along the road at 10 m, a prediction across the road at 10.5 m and an
        occluded car across the road at 11 m.
        Expected: by center the seen box takes the prediction, by corners it is 2.2 m away and
        the prediction claims the occluded box of its shape.
        Check: the mask under each cost.
        """
        masks = [
            ignored_prediction_mask(
                _long_box(10.5, pi / 2),
                torch.tensor([0.9]),
                torch.tensor([0]),
                _long_box(10.0, 0.0),
                torch.tensor([0]),
                _long_box(11.0, pi / 2),
                torch.tensor([0]),
                cost_fn=cost_fn,
                kept_claim_distance=1.0,
                claim_distance=2.0,
            ).tolist()
            for cost_fn in (cost_center, cost_corner)
        ]

        self.assertEqual(masks, [[False], [True]])

    def test_the_prediction_the_seen_box_leaves_claims_the_occluded_box(self) -> None:
        """
        Input: a seen car at 10 m, an occluded car at 12 m, predictions at 10.4 m and 11.8 m.
        Expected: the seen box takes the first prediction, the second claims the occluded box.
        Check: only the second prediction is marked.
        """
        ignored = ignored_prediction_mask(
            _boxes(10.4, 11.8),
            torch.tensor([0.9, 0.8]),
            torch.tensor([0, 0]),
            _boxes(10.0),
            torch.tensor([0]),
            _boxes(12.0),
            torch.tensor([0]),
            cost_fn=cost_center,
            kept_claim_distance=4.0,
            claim_distance=2.0,
        )

        self.assertEqual(ignored.tolist(), [False, True])

    def test_a_seen_box_keeps_a_prediction_nearer_to_an_occluded_box(self) -> None:
        """
        Input: a seen car at 10 m, an occluded car at 13.3 m, one prediction at 11.8 m.
        Expected: the seen box within its reach takes the prediction, although the occluded box
        is nearer, so a true positive is never left out.
        Check: nothing is marked.
        """
        ignored = ignored_prediction_mask(
            _boxes(11.8),
            torch.tensor([0.9]),
            torch.tensor([0]),
            _boxes(10.0),
            torch.tensor([0]),
            _boxes(13.3),
            torch.tensor([0]),
            cost_fn=cost_center,
            kept_claim_distance=4.0,
            claim_distance=2.0,
        )

        self.assertEqual(ignored.tolist(), [False])

    def test_a_prediction_of_another_class_claims_nothing(self) -> None:
        """
        Input: an occluded car and a pedestrian prediction on it.
        Expected: claims stay within a class.
        Check: nothing is marked.
        """
        ignored = ignored_prediction_mask(
            _boxes(30.2),
            torch.tensor([0.9]),
            torch.tensor([1]),
            _boxes(),
            torch.zeros(0, dtype=torch.long),
            _boxes(30.0),
            torch.tensor([0]),
            cost_fn=cost_center,
            kept_claim_distance=4.0,
            claim_distance=2.0,
        )

        self.assertEqual(ignored.tolist(), [False])


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
