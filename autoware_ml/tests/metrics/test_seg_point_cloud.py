"""Unit tests for the point-cache segmentation suite and its components.

The suite and every point-level metric are exercised on hand-built frames, so the
spatial math (boundary detection, clustering) and the suite's cache plumbing are
pinned without a model or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.segmentation3d.calibration import CalibrationError
from autoware_ml.metrics.segmentation3d.confident_error import ConfidentErrorRate
from autoware_ml.metrics.segmentation3d.entropy_auroc import UncertaintyUsefulness
from autoware_ml.metrics.segmentation3d.error_clusters import ErrorClusters
from autoware_ml.metrics.segmentation3d.partial_detection import PartialDetectionScore
from autoware_ml.metrics.segmentation3d.tolerant_error import NeighbourhoodTolerantErrorRate
from autoware_ml.metrics.segmentation3d.point_cloud import (
    FramePoints,
    PointCloudSegState,
    Segmentation3DPointCloudMetricSuite,
    confidence,
    normalized_entropy,
)

CLASS_NAMES = ("road", "obstacle")


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


def _state(frames) -> PointCloudSegState:
    return PointCloudSegState(frames, num_classes=2, ignore_index=-1, class_names=CLASS_NAMES)


def test_error_clusters_merges_nearby_errors_into_one() -> None:
    # Ten points, the first three misclassified and adjacent -> one error cluster.
    target = [0] * 10
    pred = [1, 1, 1] + [0] * 7
    out = ErrorClusters(cluster_radius=1.5).evaluate(
        _state([_frame(range(10), target, pred)]), EvalStage.TEST
    )
    assert out["error_rate"] == pytest.approx(3.0 / 10.0)
    assert out["error_cluster_count"] == pytest.approx(1.0)


def test_error_clusters_counts_singleton_error() -> None:
    # A lone misclassified point counts as one cluster (default min size 1),
    # the same penalty as a dense blob, both cause one emergency stop.
    target = [0] * 10
    pred = [1] + [0] * 9  # points are 1 m apart, radius 0.5 keeps it isolated
    out = ErrorClusters(cluster_radius=0.5).evaluate(
        _state([_frame(range(10), target, pred)]), EvalStage.TEST
    )
    assert out["error_cluster_count"] == pytest.approx(1.0)


def test_error_clusters_clean_prediction() -> None:
    target = [0] * 10
    out = ErrorClusters().evaluate(_state([_frame(range(10), target, target)]), EvalStage.TEST)
    assert out["error_rate"] == pytest.approx(0.0)
    assert out["error_cluster_count"] == pytest.approx(0.0)


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
        {
            "seg_frames": [
                {
                    "coord": torch.tensor([[float(x), 0.0, 0.0] for x in range(10)]),
                    "pred": torch.tensor(target),
                    "target": torch.tensor(target),
                    "scores": torch.full((10, 2), 0.5),
                }
            ]
        }
    )
    report = suite.result(EvalStage.TEST)
    assert "error_rate" in report
    assert "tolerant_error_rate" in report
    assert report["error_rate"] == pytest.approx(0.0)


def test_suite_demands_boxes_for_box_reading_components() -> None:
    # A frame without detection ground truth is a miswired experiment when a
    # component reads boxes, empty tensors are the explicit no-objects form.
    suite = Segmentation3DPointCloudMetricSuite(
        components=[
            PartialDetectionScore(
                box_label_to_seg_class={0: 1},
                seg_class_names=CLASS_NAMES,
                class_names=("ped",),
                stages=["test"],
            )
        ],
        num_classes=2,
        class_names=CLASS_NAMES,
    )
    frame = {
        "coord": torch.tensor([[0.0, 0.0, 0.0]]),
        "pred": torch.tensor([0]),
        "target": torch.tensor([0]),
        "scores": torch.full((1, 2), 0.5),
    }
    with pytest.raises(ValueError, match="gt_boxes and gt_box_labels"):
        suite.update({"seg_frames": [dict(frame)]})
    frame["gt_boxes"] = torch.zeros((0, 7))
    frame["gt_box_labels"] = torch.zeros((0,), dtype=torch.long)
    suite.update({"seg_frames": [frame]})


def test_calibration_error_zero_when_confidence_matches_accuracy() -> None:
    # All correct at 100% confidence -> perfectly calibrated -> ECE 0.
    target = [0, 0, 1, 1]
    scores = _scores_for(target, confidence=1.0)
    out = CalibrationError(num_bins=10).evaluate(
        _state([_frame([0, 1, 2, 3], target, target, scores=scores)]), EvalStage.TEST
    )
    assert out["ece"] == pytest.approx(0.0)


def test_calibration_error_penalizes_overconfident_mistakes() -> None:
    # Predicts confidently (95%) but is wrong half the time -> large ECE.
    target = [0, 0, 1, 1]
    pred = [0, 0, 0, 0]
    scores = _scores_for(pred, confidence=0.95)
    out = CalibrationError(num_bins=10).evaluate(
        _state([_frame([0, 1, 2, 3], target, pred, scores=scores)]), EvalStage.TEST
    )
    assert out["ece"] > 0.4


def test_entropy_auroc_perfect_separation() -> None:
    # Wrong points get high entropy (uniform), correct points get low entropy.
    target = [0, 0, 1, 1]
    pred = [0, 0, 0, 0]  # last two wrong
    scores = np.array(
        [[0.99, 0.01], [0.99, 0.01], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64
    )
    out = UncertaintyUsefulness().evaluate(
        _state([_frame([0, 1, 2, 3], target, pred, scores=scores)]), EvalStage.TEST
    )
    assert out["entropy_auroc"] == pytest.approx(1.0)


def test_confident_error_rate_flags_confident_mistakes() -> None:
    # Two misclassified points at high confidence (low entropy) -> all errors confident.
    target = [0, 0, 0, 0]
    pred = [1, 1, 0, 0]
    scores = _scores_for(pred, confidence=0.99)
    out = ConfidentErrorRate(entropy_threshold=0.3).evaluate(
        _state([_frame([0, 1, 2, 3], target, pred, scores=scores)]), EvalStage.TEST
    )
    assert out["confident_error_rate"] == pytest.approx(1.0)
    assert out["confident_error_count"] == pytest.approx(2.0)


def test_tolerant_error_forgives_correct_neighbour() -> None:
    # Point 0 (x=0) is wrong (pred 1, GT 0), point 1 (x=1) predicts 0 = point 0's true class.
    frame = _frame([0, 1], target=[0, 0], pred=[1, 0])
    state = _state([frame])
    # radius reaches the correct-class neighbour -> the wrong point is tolerated
    seen = NeighbourhoodTolerantErrorRate(radius=1.5).evaluate(state, EvalStage.TEST)
    assert seen["tolerant_error_rate"] == pytest.approx(0.0)
    assert seen["tolerant_error_count"] == pytest.approx(0.0)
    # radius too small to see it -> the wrong point survives as an error
    unseen = NeighbourhoodTolerantErrorRate(radius=0.5).evaluate(state, EvalStage.TEST)
    assert unseen["tolerant_error_rate"] == pytest.approx(0.5)
    assert unseen["tolerant_error_count"] == pytest.approx(1.0)


def test_tolerant_error_radius_zero_is_strict_error_rate() -> None:
    frame = _frame([0, 1], target=[0, 0], pred=[1, 0])
    out = NeighbourhoodTolerantErrorRate(radius=0.0).evaluate(_state([frame]), EvalStage.TEST)
    assert out["tolerant_error_rate"] == pytest.approx(0.5)  # one wrong of two, no tolerance


def test_tolerant_error_emits_global_and_per_class() -> None:
    # Points far apart (no neighbour rescue): road (class 0) has 1 wrong of 2,
    # obstacle (class 1) 0 of 2. Both global and every class are logged.
    frame = _frame([0, 10, 20, 30], target=[0, 0, 1, 1], pred=[1, 0, 1, 1])
    out = NeighbourhoodTolerantErrorRate(radius=0.5).evaluate(_state([frame]), EvalStage.TEST)
    assert out["tolerant_error_rate"] == pytest.approx(0.25)
    assert out["tolerant_error_rate_road"] == pytest.approx(0.5)
    assert out["tolerant_error_rate_obstacle"] == pytest.approx(0.0)


def test_error_clusters_emits_global_and_per_class() -> None:
    # Road: 2 adjacent wrong points -> one cluster, obstacle points all correct.
    frame = _frame([0, 1, 20, 30], target=[0, 0, 1, 1], pred=[1, 1, 1, 1])
    out = ErrorClusters(cluster_radius=1.5).evaluate(_state([frame]), EvalStage.TEST)
    assert out["error_rate"] == pytest.approx(0.5)
    assert out["error_rate_road"] == pytest.approx(1.0)
    assert out["error_cluster_count_road"] == pytest.approx(1.0)
    assert out["error_rate_obstacle"] == pytest.approx(0.0)
    assert out["error_cluster_count_obstacle"] == pytest.approx(0.0)


def test_partial_detection_credit_landmarks() -> None:
    # One box (det label 0 -> seg class 1) containing 4 points, 1 correct -> credit ~0.5.
    coord = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    scores = np.full((4, 2), 0.5)
    frame = FramePoints(
        coord=coord,
        pred=np.array([1, 0, 0, 0], dtype=np.int64),
        target=np.array([1, 1, 1, 1], dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        gt_box_labels=np.array([0], dtype=np.int64),
    )
    out = PartialDetectionScore(
        box_label_to_seg_class={0: 1}, seg_class_names=CLASS_NAMES, class_names=("ped",)
    ).evaluate(_state([frame]), EvalStage.TEST)
    # s(1)/s(4) with h=1 -> (1/2)/(4/5) = 0.625
    assert out["pd_score_ped"] == pytest.approx(0.625)
    assert out["pd_skipped_low_point_boxes"] == pytest.approx(0.0)


def test_partial_detection_ignores_points_outside_the_box_height() -> None:
    # A return from an object above the box shares its footprint but is not the
    # annotated object's point, so it must change neither n nor the credit.
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 5.0]]
    )
    scores = np.full((5, 2), 0.5)
    frame = FramePoints(
        coord=coord,
        pred=np.array([1, 0, 0, 0, 1], dtype=np.int64),
        target=np.array([1, 1, 1, 1, 1], dtype=np.int64),
        confidence=confidence(scores),
        entropy=normalized_entropy(scores),
        gt_boxes=np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        gt_box_labels=np.array([0], dtype=np.int64),
    )
    out = PartialDetectionScore(
        box_label_to_seg_class={0: 1}, seg_class_names=CLASS_NAMES, class_names=("ped",)
    ).evaluate(_state([frame]), EvalStage.TEST)
    assert out["pd_score_ped"] == pytest.approx(0.625)


def test_partial_detection_rejects_nonpositive_half_saturation() -> None:
    # h = 0 divides by zero for a box without a single correct point.
    with pytest.raises(ValueError, match="half_saturation"):
        PartialDetectionScore(
            box_label_to_seg_class={0: 1}, seg_class_names=CLASS_NAMES, half_saturation=0.0
        )


def test_partial_detection_rejects_mismatched_class_space() -> None:
    # A grouped suite folds labels into another index space, the mapping must
    # refuse to score it rather than hit the wrong classes.
    frame = _frame([0, 1], target=[0, 0], pred=[0, 0])
    metric = PartialDetectionScore(
        box_label_to_seg_class={0: 1}, seg_class_names=("other", "space"), class_names=("ped",)
    )
    with pytest.raises(ValueError, match="index space"):
        metric.evaluate(_state([frame]), EvalStage.TEST)


def test_tolerant_error_mask_equals_bruteforce_reference() -> None:
    # The per-class nearest-neighbour formulation must reproduce the direct
    # definition: wrong, and no point predicted the true class within radius.
    from autoware_ml.metrics.segmentation3d.spatial import tolerant_error_mask

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


def test_entropy_auroc_binned_matches_rankdata_reference() -> None:
    # The histogram closed form must agree with exact tie-aware ranking to well
    # under one bin width.
    import numpy as np
    from scipy.stats import rankdata

    from autoware_ml.metrics.segmentation3d.entropy_auroc import binned_auroc

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
