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

"""Unit tests for the point-cache segmentation suite and its components.

The suite and every point-level metric are exercised on hand-built frames, so the
spatial math (boundary detection, clustering) and the suite's cache plumbing are
pinned without a model or GPU.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.segmentation3d.calibration import CalibrationError
from autoware_ml.metrics.segmentation3d.confident_error import ConfidentErrorRate
from autoware_ml.metrics.segmentation3d.entropy_auroc import UncertaintyUsefulness, binned_auroc
from autoware_ml.metrics.segmentation3d.error_clusters import ErrorClusters
from autoware_ml.metrics.segmentation3d.frame_count import FrameCount
from autoware_ml.metrics.segmentation3d.partial_detection3d import PartialDetection3DScore
from autoware_ml.metrics.segmentation3d.tolerant_error import NeighbourhoodTolerantErrorRate
from autoware_ml.metrics.segmentation3d.point_cloud import (
    FramePoints,
    PointCloudSeg3DState,
    Segmentation3DPointCloudMetricSuite,
    confidence,
    normalized_entropy,
)
from autoware_ml.metrics.segmentation3d.spatial import tolerant_error_mask
from autoware_ml.metrics.report import MetricKey
from autoware_ml.metrics.tests.eval_output_fixtures import (
    segmentation_eval_output,
    segmentation_frame,
)

CLASS_NAMES = ("road", "obstacle")
# The partial detection score joins the taxonomies by a shared prefix: one
# detection class, and a segmentation taxonomy that starts with it.
DET3D_CLASS_NAMES = ("ped",)
SEG3D_CLASS_NAMES = ("ped", "road")


def _scores_for(pred, confidence: float, num_classes: int = 2) -> np.ndarray:
    """Softmax rows whose argmax is ``pred`` at the given max probability."""
    rest = (1.0 - confidence) / (num_classes - 1)
    scores = np.full((len(pred), num_classes), rest, dtype=np.float64)
    scores[np.arange(len(pred)), np.array(pred)] = confidence
    return scores


def _frame(xs, target, pred, num_classes: int = 2, scores=None) -> FramePoints:
    coord = np.array([[float(x), 0.0, 0.0] for x in xs], dtype=np.float64)
    if scores is None:
        scores = np.full((len(xs), num_classes), 1.0 / num_classes, dtype=np.float64)
    return FramePoints(
        coord=coord,
        pred=np.array(pred, dtype=np.int64),
        target=np.array(target, dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.zeros((0, 7), dtype=np.float64),
        gt_box_labels=np.zeros((0,), dtype=np.int64),
    )


def _state(frames, class_names=CLASS_NAMES) -> PointCloudSeg3DState:
    return PointCloudSeg3DState(
        frames, num_classes=2, ignore_index=-1, class_names=class_names, num_frames=len(frames)
    )


def test_error_clusters_merges_nearby_errors_into_one() -> None:
    # Ten points, the first three misclassified and adjacent -> one error cluster.
    target = [0] * 10
    pred = [1, 1, 1] + [0] * 7
    out = ErrorClusters(cluster_radius=1.5).evaluate(
        _state([_frame(range(10), target, pred)]), EvalStage.TEST
    )
    assert out.metric_value(MetricKey(name="error_rate")) == pytest.approx(3.0 / 10.0)
    assert out.metric_value(MetricKey(name="error_cluster_count")) == pytest.approx(1.0)


def test_error_clusters_counts_singleton_error() -> None:
    # A lone misclassified point counts as one cluster (default min size 1),
    # the same penalty as a dense blob, both cause one emergency stop.
    target = [0] * 10
    pred = [1] + [0] * 9  # points are 1 m apart, radius 0.5 keeps it isolated
    out = ErrorClusters(cluster_radius=0.5).evaluate(
        _state([_frame(range(10), target, pred)]), EvalStage.TEST
    )
    assert out.metric_value(MetricKey(name="error_cluster_count")) == pytest.approx(1.0)


def test_error_clusters_clean_prediction() -> None:
    target = [0] * 10
    out = ErrorClusters().evaluate(_state([_frame(range(10), target, target)]), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="error_rate")) == pytest.approx(0.0)
    assert out.metric_value(MetricKey(name="error_cluster_count")) == pytest.approx(0.0)


def test_suite_caches_frames_and_runs_components() -> None:
    target = [0] * 5 + [1] * 5
    suite = Segmentation3DPointCloudMetricSuite(
        components=[
            ErrorClusters(stages=["test"]),
            NeighbourhoodTolerantErrorRate(stages=["test"]),
        ],
        num_classes=2,
        class_names=CLASS_NAMES,
    )
    suite.update(
        segmentation_eval_output(
            [
                segmentation_frame(
                    coord=torch.tensor([[float(x), 0.0, 0.0] for x in range(10)]),
                    pred=torch.tensor(target),
                    target=torch.tensor(target),
                    scores=torch.full((10, 2), 0.5),
                )
            ]
        )
    )
    report = suite.result(EvalStage.TEST)
    assert MetricKey(name="error_rate") in report.metric_names()
    assert MetricKey(name="tolerant_error_rate") in report.metric_names()
    assert report.metric_value(MetricKey(name="error_rate")) == pytest.approx(0.0)


def test_suite_frame_count_reads_the_cache_length() -> None:
    suite = Segmentation3DPointCloudMetricSuite(
        components=[FrameCount(stages=["test"])],
        num_classes=2,
        class_names=CLASS_NAMES,
    )
    frame = segmentation_frame(
        coord=torch.tensor([[1.0, 0.0, 0.0]]),
        pred=torch.tensor([0]),
        target=torch.tensor([0]),
        scores=torch.full((1, 2), 0.5),
    )
    suite.update(segmentation_eval_output([frame, frame]))
    suite.update(segmentation_eval_output([frame]))

    report = suite.result(EvalStage.TEST)

    assert report.metric_value(MetricKey(name="num_frames")) == 3.0
    # The count is range-invariant: reported once per slice, not per window.
    assert MetricKey(name="num_frames", range_suffix="50m_90m") not in report.metric_names()


def test_suite_demands_boxes_for_box_reading_components() -> None:
    # A frame without detection ground truth is a miswired experiment when a
    # component reads boxes, empty tensors are the explicit no-objects form.
    suite = Segmentation3DPointCloudMetricSuite(
        components=[
            PartialDetection3DScore(
                det3d_class_names=DET3D_CLASS_NAMES,
                seg3d_class_names=SEG3D_CLASS_NAMES,
                class_names=("ped",),
                stages=["test"],
            )
        ],
        num_classes=2,
        class_names=SEG3D_CLASS_NAMES,
    )
    without_boxes = segmentation_frame(
        coord=torch.tensor([[0.0, 0.0, 0.0]]),
        pred=torch.tensor([0]),
        target=torch.tensor([0]),
        scores=torch.full((1, 2), 0.5),
    )
    with pytest.raises(ValueError, match="gt_boxes and gt_box_labels"):
        suite.update(segmentation_eval_output([without_boxes]))
    with_empty_boxes = segmentation_frame(
        coord=torch.tensor([[0.0, 0.0, 0.0]]),
        pred=torch.tensor([0]),
        target=torch.tensor([0]),
        scores=torch.full((1, 2), 0.5),
        gt_boxes=torch.zeros((0, 7)),
        gt_box_labels=torch.zeros((0,), dtype=torch.long),
    )
    suite.update(segmentation_eval_output([with_empty_boxes]))


def test_calibration_error_zero_when_confidence_matches_accuracy() -> None:
    # All correct at 100% confidence -> perfectly calibrated -> ECE 0.
    target = [0, 0, 1, 1]
    scores = _scores_for(target, confidence=1.0)
    out = CalibrationError(num_bins=10).evaluate(
        _state([_frame([0, 1, 2, 3], target, target, scores=scores)]), EvalStage.TEST
    )
    assert out.metric_value(MetricKey(name="ece")) == pytest.approx(0.0)


def test_calibration_error_penalizes_overconfident_mistakes() -> None:
    # Predicts confidently (95%) but is wrong half the time -> large ECE.
    target = [0, 0, 1, 1]
    pred = [0, 0, 0, 0]
    scores = _scores_for(pred, confidence=0.95)
    out = CalibrationError(num_bins=10).evaluate(
        _state([_frame([0, 1, 2, 3], target, pred, scores=scores)]), EvalStage.TEST
    )
    assert out.metric_value(MetricKey(name="ece")) > 0.4


def test_entropy_auroc_perfect_separation() -> None:
    # Wrong points get high entropy (uniform), correct points get low entropy.
    target = [0, 0, 1, 1]
    pred = [0, 0, 0, 0]  # last two wrong
    scores = np.array([[0.99, 0.01], [0.99, 0.01], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    out = UncertaintyUsefulness().evaluate(
        _state([_frame([0, 1, 2, 3], target, pred, scores=scores)]), EvalStage.TEST
    )
    assert out.metric_value(MetricKey(name="entropy_auroc")) == pytest.approx(1.0)


def test_confident_error_rate_flags_confident_mistakes() -> None:
    # Two misclassified points at high confidence (low entropy) -> all errors confident.
    target = [0, 0, 0, 0]
    pred = [1, 1, 0, 0]
    scores = _scores_for(pred, confidence=0.99)
    out = ConfidentErrorRate(entropy_threshold=0.3).evaluate(
        _state([_frame([0, 1, 2, 3], target, pred, scores=scores)]), EvalStage.TEST
    )
    assert out.metric_value(MetricKey(name="confident_error_rate")) == pytest.approx(1.0)
    assert out.metric_value(MetricKey(name="confident_error_count")) == pytest.approx(2.0)


def test_tolerant_error_forgives_correct_neighbour() -> None:
    # Point 0 (x=0) is wrong (pred 1, GT 0), point 1 (x=1) predicts 0 = point 0's true class.
    frame = _frame([0, 1], target=[0, 0], pred=[1, 0])
    state = _state([frame])
    # radius reaches the correct-class neighbour -> the wrong point is tolerated
    seen = NeighbourhoodTolerantErrorRate(radius=1.5).evaluate(state, EvalStage.TEST)
    assert seen.metric_value(MetricKey(name="tolerant_error_rate")) == pytest.approx(0.0)
    assert seen.metric_value(MetricKey(name="tolerant_error_count")) == pytest.approx(0.0)
    # radius too small to see it -> the wrong point survives as an error
    unseen = NeighbourhoodTolerantErrorRate(radius=0.5).evaluate(state, EvalStage.TEST)
    assert unseen.metric_value(MetricKey(name="tolerant_error_rate")) == pytest.approx(0.5)
    assert unseen.metric_value(MetricKey(name="tolerant_error_count")) == pytest.approx(1.0)


def test_tolerant_error_radius_zero_is_strict_error_rate() -> None:
    frame = _frame([0, 1], target=[0, 0], pred=[1, 0])
    out = NeighbourhoodTolerantErrorRate(radius=0.0).evaluate(_state([frame]), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="tolerant_error_rate")) == pytest.approx(
        0.5
    )  # one wrong of two, no tolerance


def test_tolerant_error_emits_global_and_per_class() -> None:
    # Points far apart (no neighbour rescue): road (class 0) has 1 wrong of 2,
    # obstacle (class 1) 0 of 2. Both global and every class are logged.
    frame = _frame([0, 10, 20, 30], target=[0, 0, 1, 1], pred=[1, 0, 1, 1])
    out = NeighbourhoodTolerantErrorRate(radius=0.5).evaluate(_state([frame]), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="tolerant_error_rate")) == pytest.approx(0.25)
    assert out.metric_value(MetricKey(name="tolerant_error_rate_road")) == pytest.approx(0.5)
    assert out.metric_value(MetricKey(name="tolerant_error_rate_obstacle")) == pytest.approx(0.0)


def test_error_clusters_emits_global_and_per_class() -> None:
    # Road: 2 adjacent wrong points -> one cluster, obstacle points all correct.
    frame = _frame([0, 1, 20, 30], target=[0, 0, 1, 1], pred=[1, 1, 1, 1])
    out = ErrorClusters(cluster_radius=1.5).evaluate(_state([frame]), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="error_rate")) == pytest.approx(0.5)
    assert out.metric_value(MetricKey(name="error_rate_road")) == pytest.approx(1.0)
    assert out.metric_value(MetricKey(name="error_cluster_count_road")) == pytest.approx(1.0)
    assert out.metric_value(MetricKey(name="error_rate_obstacle")) == pytest.approx(0.0)
    assert out.metric_value(MetricKey(name="error_cluster_count_obstacle")) == pytest.approx(0.0)


def test_partial_detection3d_credit_landmarks() -> None:
    # One box (det class 0, the shared index) containing 4 points, 1 correct -> credit ~0.5.
    coord = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    scores = np.full((4, 2), 0.5)
    frame = FramePoints(
        coord=coord,
        pred=np.array([0, 1, 1, 1], dtype=np.int64),
        target=np.array([0, 0, 0, 0], dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        gt_box_labels=np.array([0], dtype=np.int64),
    )
    out = PartialDetection3DScore(
        det3d_class_names=DET3D_CLASS_NAMES,
        seg3d_class_names=SEG3D_CLASS_NAMES,
        class_names=("ped",),
    ).evaluate(_state([frame], SEG3D_CLASS_NAMES), EvalStage.TEST)
    # s(1)/s(4) with h=1 -> (1/2)/(4/5) = 0.625
    assert out.metric_value(MetricKey(name="pd_score_ped")) == pytest.approx(0.625)
    assert out.metric_value(MetricKey(name="pd_skipped_low_point_boxes")) == pytest.approx(0.0)


def test_partial_detection3d_ignores_points_outside_the_box_height() -> None:
    # A return from an object above the box shares its footprint but is not the
    # annotated object's point, so it must change neither n nor the credit.
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 5.0]]
    )
    scores = np.full((5, 2), 0.5)
    frame = FramePoints(
        coord=coord,
        pred=np.array([0, 1, 1, 1, 0], dtype=np.int64),
        target=np.array([0, 0, 0, 0, 0], dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        gt_box_labels=np.array([0], dtype=np.int64),
    )
    out = PartialDetection3DScore(
        det3d_class_names=DET3D_CLASS_NAMES,
        seg3d_class_names=SEG3D_CLASS_NAMES,
        class_names=("ped",),
    ).evaluate(_state([frame], SEG3D_CLASS_NAMES), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="pd_score_ped")) == pytest.approx(0.625)


def test_partial_detection3d_scores_only_the_objects_own_points() -> None:
    # A pedestrian box reaches the ground, so ground returns share its footprint. They
    # carry the ground target, so neither the count nor the credit may see them: all
    # four pedestrian points correct is a full credit, whatever the ground does.
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.3, 0.3, -0.9],
            [-0.3, -0.3, -0.9],
        ]
    )
    scores = np.full((6, 2), 0.5)
    frame = FramePoints(
        coord=coord,
        pred=np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
        target=np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        gt_box_labels=np.array([0], dtype=np.int64),
    )
    out = PartialDetection3DScore(
        det3d_class_names=DET3D_CLASS_NAMES,
        seg3d_class_names=SEG3D_CLASS_NAMES,
        class_names=("ped",),
    ).evaluate(_state([frame], SEG3D_CLASS_NAMES), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="pd_score_ped")) == pytest.approx(1.0)


def test_partial_detection3d_does_not_credit_ground_predicted_as_the_object() -> None:
    # None of the pedestrian's own points were found, and the ground inside the box was
    # mispredicted as pedestrian. The box scores zero, the ground cannot stand in for it.
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.3, -0.9],
            [-0.3, -0.3, -0.9],
        ]
    )
    scores = np.full((4, 2), 0.5)
    frame = FramePoints(
        coord=coord,
        pred=np.array([1, 1, 0, 0], dtype=np.int64),
        target=np.array([0, 0, 1, 1], dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        gt_box_labels=np.array([0], dtype=np.int64),
    )
    out = PartialDetection3DScore(
        det3d_class_names=DET3D_CLASS_NAMES,
        seg3d_class_names=SEG3D_CLASS_NAMES,
        class_names=("ped",),
    ).evaluate(_state([frame], SEG3D_CLASS_NAMES), EvalStage.TEST)
    assert out.metric_value(MetricKey(name="pd_score_ped")) == pytest.approx(0.0)


def test_partial_detection3d_rejects_non_positive_half_saturation() -> None:
    # h = 0 divides by zero for a box without a single correct point.
    with pytest.raises(ValueError, match="half_saturation"):
        PartialDetection3DScore(
            det3d_class_names=DET3D_CLASS_NAMES,
            seg3d_class_names=SEG3D_CLASS_NAMES,
            class_names=("ped",),
            half_saturation=0.0,
        )


def test_partial_detection3d_rejects_mismatched_class_space() -> None:
    # A grouped suite folds labels into another index space, the shared index
    # must refuse to score it rather than hit the wrong classes.
    frame = _frame([0, 1], target=[0, 0], pred=[0, 0])
    metric = PartialDetection3DScore(
        det3d_class_names=DET3D_CLASS_NAMES,
        seg3d_class_names=SEG3D_CLASS_NAMES,
        class_names=("ped",),
    )
    with pytest.raises(ValueError, match="index space"):
        metric.evaluate(_state([frame]), EvalStage.TEST)


def test_partial_detection3d_requires_the_shared_taxonomy_prefix() -> None:
    # The score relies on a detection class and its segmentation class sharing an
    # index, so a segmentation taxonomy that does not start with the detection one
    # is refused up front, and so is a scored class the detection taxonomy lacks.
    with pytest.raises(ValueError, match="must start with"):
        PartialDetection3DScore(
            det3d_class_names=("ped",), seg3d_class_names=("road", "ped"), class_names=("ped",)
        )
    with pytest.raises(ValueError, match="not detection classes"):
        PartialDetection3DScore(
            det3d_class_names=DET3D_CLASS_NAMES,
            seg3d_class_names=SEG3D_CLASS_NAMES,
            class_names=("car",),
        )


def test_tolerant_error_mask_equals_brute_force_reference() -> None:
    # The per-class nearest-neighbour formulation must reproduce the direct
    # definition: wrong, and no point predicted the true class within radius.
    rng = np.random.default_rng(3)
    coord = rng.uniform(-20, 20, (2000, 3))
    target = rng.integers(0, 3, 2000)
    pred = target.copy()
    flip = rng.random(2000) < 0.15
    pred[flip] = rng.integers(0, 3, int(flip.sum()))

    def _reference(radius: float) -> np.ndarray:
        wrong = pred != target
        mask = np.zeros(coord.shape[0], dtype=bool)
        for index in np.flatnonzero(wrong):
            distance = np.linalg.norm(coord - coord[index], axis=1)
            rescued = np.any((distance <= radius) & (pred == target[index]))
            mask[index] = not rescued
        return mask

    for radius in (0.0, 0.8, 3.0):
        np.testing.assert_array_equal(
            tolerant_error_mask(coord, pred, target, radius), _reference(radius)
        )


class TestTolerantErrorMask(unittest.TestCase):
    """Tests the neighbourhood rescue at and around its radius and with a stray point."""

    def test_rescues_at_exactly_the_radius(self) -> None:
        """
        Input: two wrong points, each the other's only candidate of its true class, once
        exactly 0.25 m apart and once one float step further, with a 0.25 m radius.
        Expected: a candidate at exactly the radius rescues, as the counting query did, and
        one a step beyond does not.
        Check: no error at the radius, both points errors one step beyond it.
        """
        target = np.array([0, 1])
        pred = np.array([1, 0])
        at_radius = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        beyond = np.array([[0.0, 0.0, 0.0], [np.nextafter(0.25, 1.0), 0.0, 0.0]])

        np.testing.assert_array_equal(
            tolerant_error_mask(at_radius, pred, target, 0.25), [False, False]
        )
        np.testing.assert_array_equal(tolerant_error_mask(beyond, pred, target, 0.25), [True, True])

    def test_matches_the_counting_query_within_rounding_of_the_radius(self) -> None:
        """
        Input: 20,000 pairs of wrong points, each point the other's candidate, spread over
        200 m, with the two points of a pair a radius apart give or take up to four float
        steps in a random direction, for a 1 m radius.
        Expected: every pair whose distance rounds to within the radius is rescued exactly
        as the previous counting query (``query_ball_point``) rescued it, so the distance
        check must use the query's own arithmetic.
        Check: the mask equals the counting query's mask, and the pairs include both
        rescued and unrescued points.
        """
        rng = np.random.default_rng(7)
        radius, pairs = 1.0, 20_000
        first = rng.uniform(-100.0, 100.0, (pairs, 3))
        direction = rng.normal(size=(pairs, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        steps = rng.integers(-4, 5, pairs) * np.finfo(np.float64).eps
        second = first + direction * (radius * (1.0 + steps))[:, None]
        coord = np.concatenate([first, second])
        target = np.repeat([0, 1], pairs)
        pred = np.repeat([1, 0], pairs)

        expected = np.ones(2 * pairs, dtype=bool)
        for true_class in (0, 1):
            queries = np.flatnonzero(target == true_class)
            counts = cKDTree(coord[pred == true_class]).query_ball_point(
                coord[queries], r=radius, return_length=True
            )
            expected[queries[counts > 0]] = False
        self.assertTrue(expected.any() and not expected.all())

        np.testing.assert_array_equal(tolerant_error_mask(coord, pred, target, radius), expected)

    def test_ignores_a_point_far_from_the_scene(self) -> None:
        """
        Input: 600 random points with some errors, then the same points plus one correct
        point 1e9 m away, of a class no other point has.
        Expected: the stray point makes the candidate grid too large, so the search falls
        back to every candidate and still gives the same mask; the stray point rescues no
        one.
        Check: the mask of the 600 points is unchanged and the stray point is no error.
        """
        rng = np.random.default_rng(11)
        coord = rng.uniform(-5, 5, (600, 3))
        target = rng.integers(0, 4, 600)
        pred = np.where(rng.random(600) < 0.3, rng.integers(0, 4, 600), target)
        expected = tolerant_error_mask(coord, pred, target, 0.6)
        self.assertTrue(expected.any() and not expected.all())

        far = np.vstack([coord, [[1.0e9, 0.0, 0.0]]])
        mask = tolerant_error_mask(far, np.append(pred, 9), np.append(target, 9), 0.6)

        np.testing.assert_array_equal(mask[:600], expected)
        self.assertFalse(mask[600])


def test_entropy_auroc_binned_matches_rankdata_reference() -> None:
    # The histogram closed form must agree with exact tie-aware ranking to well
    # under one bin width.
    rng = np.random.default_rng(5)
    entropy = rng.beta(2.0, 5.0, 50_000)
    wrong = rng.random(50_000) < 0.2
    entropy[wrong] += 0.15  # wrong points shifted toward higher entropy
    entropy = np.clip(entropy, 0.0, 1.0)

    ranks = rankdata(entropy)
    n_pos, n_neg = int(wrong.sum()), int((~wrong).sum())
    exact = (ranks[wrong].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    num_bins = 8192
    bins = np.minimum((entropy * num_bins).astype(np.int64), num_bins - 1)
    wrong_hist = np.bincount(bins[wrong], minlength=num_bins).astype(np.float64)
    correct_hist = np.bincount(bins[~wrong], minlength=num_bins).astype(np.float64)
    assert abs(binned_auroc(wrong_hist, correct_hist) - exact) < 1e-3
