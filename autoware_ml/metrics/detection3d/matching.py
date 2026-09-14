"""Matching math and the memoizing detection state shared by the metrics.

Metrics ask a :class:`DetectionState` for match curves and turn them into AP,
APH, NDS, or TP errors. The state memoizes per class and threshold, so the
expensive matching runs once and is shared across every metric. Matching is
greedy and score-ordered with a configurable cost: nuScenes-style BEV center
distance by default, corner distance optionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np
import torch

from autoware_ml.metrics.base import MetricRange
from autoware_ml.metrics.detection3d.geometry import (
    corner_displacement_matrix,
    corner_displacements,
    nearest_surface_distances,
)
from autoware_ml.metrics.detection3d.structures import (
    ERROR_NAMES,
    CurveMetrics,
    Detection3DSample,
    MatchCurve,
    SelectedTpErrors,
)


def cost_center(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """BEV center distance for every prediction and ground-truth pair, ``(P, G)``.

    Args:
        pred_boxes: Prediction box rows ``(P, 7+)``.
        gt_boxes: Ground-truth box rows ``(G, 7+)``.

    Returns:
        The ``(P, G)`` cost matrix in meters.
    """
    return np.linalg.norm(pred_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=2)


def cost_corner(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Mean BEV corner distance for every prediction and ground-truth pair, ``(P, G)``.

    Args:
        pred_boxes: Prediction box rows ``(P, 7+)``.
        gt_boxes: Ground-truth box rows ``(G, 7+)``.

    Returns:
        The ``(P, G)`` cost matrix in meters.
    """
    return corner_displacement_matrix(pred_boxes, gt_boxes)


MATCH_COSTS = {"center": cost_center, "corner": cost_corner}


def resolve_match_cost(name: str):
    """Return the matching cost function for a configured name (fail-loud).

    Args:
        name: Configured cost name, ``center`` or ``corner``.

    Returns:
        The cost function computing a ``(P, G)`` matrix.
    """
    if name not in MATCH_COSTS:
        raise ValueError(f"Unknown match cost {name!r}, expected one of {sorted(MATCH_COSTS)}.")
    return MATCH_COSTS[name]


def _distance_mask(
    boxes: torch.Tensor,
    min_distance: float,
    max_distance: float | None,
) -> torch.Tensor:
    """Boolean mask selecting boxes within a radial distance window (XY norm)."""
    if boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)

    distances = torch.linalg.vector_norm(boxes[:, :2].to(dtype=torch.float32), dim=1)
    mask = distances >= min_distance
    if max_distance is not None:
        mask &= distances < max_distance
    return mask


def gt_keep_mask(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_num_points: torch.Tensor | None,
    class_names: tuple[str, ...],
    eval_class_range: dict[str, float] | None,
    min_num_points: int,
) -> torch.Tensor:
    """Build the per-frame GT keep mask applied at accumulation time.

    Combines per-class distance caps and the minimum LiDAR point count into one
    boolean mask, so the suite stores only kept GT.

    Args:
        gt_boxes: Ground-truth box tensor ``(G, 7+)``.
        gt_labels: Integer class labels ``(G,)``.
        gt_num_points: Lidar points per box, or ``None`` to skip the point-count filter.
        class_names: Class names in label order.
        eval_class_range: Class name to maximum evaluation distance in meters.
        min_num_points: Minimum lidar points for a box to count.

    Returns:
        Boolean keep mask ``(G,)``.
    """
    n = gt_boxes.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=gt_boxes.device)
    if n == 0:
        return keep

    if eval_class_range:
        distances = torch.linalg.vector_norm(gt_boxes[:, :2].to(dtype=torch.float32), dim=1)
        for box_idx in range(n):
            label = int(gt_labels[box_idx].item())
            if 0 <= label < len(class_names):
                max_dist = eval_class_range.get(class_names[label])
                if max_dist is not None and float(distances[box_idx].item()) > max_dist:
                    keep[box_idx] = False

    if min_num_points > 0 and gt_num_points is not None:
        keep &= gt_num_points >= min_num_points

    return keep


def _slice_sample(
    sample: Detection3DSample,
    gt_keep: torch.Tensor,
    pred_keep: torch.Tensor,
) -> Detection3DSample:
    """Return a new sample with GT and predictions sliced by boolean masks."""
    return Detection3DSample(
        pred_boxes=sample.pred_boxes[pred_keep],
        pred_scores=sample.pred_scores[pred_keep],
        pred_labels=sample.pred_labels[pred_keep],
        gt_boxes=sample.gt_boxes[gt_keep],
        gt_labels=sample.gt_labels[gt_keep],
        gt_ttc=None if sample.gt_ttc is None else sample.gt_ttc[gt_keep],
        pred_ttc=None if sample.pred_ttc is None else sample.pred_ttc[pred_keep],
        ttc_covered=sample.ttc_covered,
    )


def clip_to_range(
    samples: list[Detection3DSample],
    metric_range: MetricRange,
) -> list[Detection3DSample]:
    """Clip both GT and predictions to a radial distance window.

    Args:
        samples: Accumulated per-frame samples.
        metric_range: Radial window in meters.

    Returns:
        New samples with boxes outside the window removed.
    """
    return [
        _slice_sample(
            sample,
            gt_keep=_distance_mask(
                sample.gt_boxes, metric_range.min_distance, metric_range.max_distance
            ),
            pred_keep=_distance_mask(
                sample.pred_boxes, metric_range.min_distance, metric_range.max_distance
            ),
        )
        for sample in samples
    ]


def labels_to_evaluate(
    samples: list[Detection3DSample],
    class_names: tuple[str, ...] | None,
) -> list[int]:
    """All class indices when ``class_names`` is given, else only present labels.

    Args:
        samples: Accumulated per-frame samples.
        class_names: Class names in label order, or ``None``.

    Returns:
        The class indices to evaluate.
    """
    if class_names is not None:
        return list(range(len(class_names)))
    if not samples:
        return []
    labels = torch.unique(torch.cat([sample.gt_labels.reshape(-1) for sample in samples]))
    return [int(label) for label in labels.tolist() if int(label) >= 0]


class PreparedLabel:
    """One class's boxes across all frames, flattened in global score order.

    Everything threshold-independent is done once here (masking, tensor to
    NumPy conversion, the per-frame cost matrices, and the global score sort),
    so :class:`DetectionState` builds this once per label and matches every
    threshold against it. Per-frame boxes are concatenated in frame order and
    stably sorted by descending score. Greedy claims never cross frames, so
    the flat order matches per-frame greedy matching exactly.
    """

    __slots__ = (
        "total_gt",
        "pred_boxes",
        "scores",
        "cost_rows",
        "candidate_rows",
        "gt_flat",
        "gt_flat_index",
    )

    def __init__(self, samples: list[Detection3DSample], label: int, cost_fn) -> None:
        """Mask, convert and cost the label's boxes across all frames, in score order.

        Args:
            samples: Per-frame detection samples.
            label: Class index this preparation is for.
            cost_fn: ``cost_fn(pred_boxes, gt_boxes)`` returning the ``(P, G)`` cost matrix.
        """
        pred_chunks: list[np.ndarray] = []
        score_chunks: list[np.ndarray] = []
        cost_row_chunks: list[list[np.ndarray]] = []
        candidate_row_chunks: list[list[np.ndarray]] = []
        gt_index_chunks: list[np.ndarray] = []
        gt_frames: list[np.ndarray] = []
        self.total_gt = 0
        gt_offset = 0
        for sample in samples:
            _validate_box_tensor(sample.gt_boxes, "gt_boxes")
            _validate_box_tensor(sample.pred_boxes, "pred_boxes")
            gt_boxes = sample.gt_boxes[sample.gt_labels == label].numpy().astype(np.float64)
            self.total_gt += int(gt_boxes.shape[0])
            pred_mask = sample.pred_labels == label
            if not bool(pred_mask.any()):
                continue
            pred_boxes = sample.pred_boxes[pred_mask].numpy().astype(np.float64)
            cost = cost_fn(pred_boxes, gt_boxes)  # (P, G)
            pred_chunks.append(pred_boxes)
            score_chunks.append(sample.pred_scores[pred_mask].numpy().astype(np.float64))
            cost_row_chunks.append(list(cost))
            # Candidates in ascending cost order, once per row: matching at any
            # threshold then only walks each row past its claimed entries. The
            # stable sort keeps the lowest index first among equal costs, the
            # argmin tie rule.
            candidate_row_chunks.append(np.argsort(cost, axis=1, kind="stable").tolist())
            gt_frames.append(gt_boxes)
            gt_index_chunks.append(np.full(pred_boxes.shape[0], gt_offset, dtype=np.int64))
            gt_offset += gt_boxes.shape[0]

        if not pred_chunks:
            self.pred_boxes = np.zeros((0, 7), dtype=np.float64)
            self.scores = np.zeros(0, dtype=np.float64)
            self.cost_rows: list[np.ndarray] = []
            self.candidate_rows: list[list[int]] = []
            self.gt_flat = np.zeros((0, 7), dtype=np.float64)
            self.gt_flat_index = np.zeros(0, dtype=np.int64)
            return

        scores = np.concatenate(score_chunks)
        order = np.argsort(-scores, kind="stable")
        self.pred_boxes = np.concatenate(pred_chunks)[order]
        self.scores = scores[order]
        # Per prediction: its frame's cost row and the flat offset of GT slot 0
        # in that frame, so a claim index maps straight into gt_flat.
        cost_rows = [row for chunk in cost_row_chunks for row in chunk]
        self.cost_rows = [cost_rows[index] for index in order]
        candidate_rows = [row for chunk in candidate_row_chunks for row in chunk]
        self.candidate_rows = [candidate_rows[index] for index in order]
        self.gt_flat = (
            np.concatenate(gt_frames) if gt_frames else np.zeros((0, 7), dtype=np.float64)
        )
        self.gt_flat_index = np.concatenate(gt_index_chunks)[order]

    def match(self, threshold: float) -> MatchCurve:
        """Greedy matching at one threshold over the prepared, score-ordered boxes.

        The Python loop does only the inherently sequential part, claiming,
        against precomputed cost rows. Every error channel is then computed
        vectorized over all matched pairs at once.

        Args:
            threshold: Maximum match cost in meters.

        Returns:
            The score-ordered match curve at ``threshold``.
        """
        num_pred = self.scores.shape[0]
        true_positive = np.zeros(num_pred, dtype=bool)
        matched_flat_gt = np.full(num_pred, -1, dtype=np.int64)
        claimed = np.zeros(self.gt_flat.shape[0], dtype=bool)
        for index in range(num_pred):
            costs = self.cost_rows[index]
            offset = int(self.gt_flat_index[index])
            # The first unclaimed candidate in ascending cost order is the
            # nearest unclaimed ground truth, past the threshold nothing later
            # can match either.
            for candidate_gt in self.candidate_rows[index]:
                if costs[candidate_gt] > threshold:
                    break
                flat = offset + candidate_gt
                if claimed[flat]:
                    continue
                true_positive[index] = True
                matched_flat_gt[index] = flat
                claimed[flat] = True
                break

        heading_score = np.zeros(num_pred, dtype=np.float64)
        errors = {name: np.full(num_pred, np.nan, dtype=np.float64) for name in ERROR_NAMES}
        corner_error = np.full(num_pred, np.nan, dtype=np.float64)
        nearest_surface = np.full(num_pred, np.nan, dtype=np.float64)
        matched = matched_flat_gt >= 0
        if matched.any():
            pred_boxes = self.pred_boxes[matched]
            gt_boxes = self.gt_flat[matched_flat_gt[matched]]
            errors["ATE"][matched] = np.linalg.norm(pred_boxes[:, :2] - gt_boxes[:, :2], axis=1)
            errors["AOE"][matched] = _orientation_errors(pred_boxes, gt_boxes)
            errors["ASE"][matched] = _scale_errors(pred_boxes, gt_boxes)
            errors["AVE"][matched] = _velocity_errors(pred_boxes, gt_boxes)
            # TODO: attribute errors are not evaluated, every match carries the neutral 1.0.
            errors["AAE"][matched] = 1.0
            heading_score[matched] = _heading_scores(errors["AOE"][matched])
            corner_error[matched] = corner_displacements(pred_boxes, gt_boxes)
            nearest_surface[matched] = nearest_surface_distances(
                pred_boxes
            ) - nearest_surface_distances(gt_boxes)

        return MatchCurve(
            total_gt=self.total_gt,
            scores=self.scores,
            true_positive=true_positive,
            false_positive=~true_positive,
            heading_score=heading_score,
            translation_error=errors["ATE"],
            orientation_error=errors["AOE"],
            scale_error=errors["ASE"],
            velocity_error=errors["AVE"],
            attribute_error=errors["AAE"],
            corner_error=corner_error,
            nearest_surface_error=nearest_surface,
        )


def match_by_cost(
    samples: list[Detection3DSample],
    label: int,
    threshold: float,
    cost_fn,
) -> MatchCurve:
    """Greedy score-ordered matching for one class and threshold under ``cost_fn``.

    ``cost_fn(pred_boxes, gt_boxes)`` returns the ``(P, G)`` match-cost matrix. A
    prediction matches its lowest-cost unclaimed GT in its frame when that cost is
    within ``threshold``. Center distance reproduces the nuScenes behaviour,
    corner distance couples size and yaw into the match itself. One-threshold
    form of :class:`PreparedLabel`, which callers matching several thresholds
    should build once and reuse.

    Args:
        samples: Accumulated per-frame samples.
        label: Class label to match.
        threshold: Maximum match cost in meters.
        cost_fn: Cost function computing the ``(P, G)`` matrix.

    Returns:
        The score-ordered match curve.
    """
    return PreparedLabel(samples, label, cost_fn).match(threshold)


def _validate_box_tensor(boxes: torch.Tensor, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"{name} must have shape (N, 7+) but got {tuple(boxes.shape)}.")


def curve_metrics(curve: MatchCurve) -> CurveMetrics:
    """AP, APH, max-F1 and the optimal-confidence operating point for a curve.

    Callers holding a :class:`DetectionState` should use its memoized
    ``curve_metrics(label, threshold)`` instead of calling this repeatedly.

    Args:
        curve: Score-ordered match curve.

    Returns:
        The curve summary.
    """
    cumulative_fp = curve.cumulative_fp  # cumsum once, both curves share it
    precision, recall = _precision_recall(curve.cumulative_tp, cumulative_fp, curve.total_gt)
    ap = _interpolated_ap(precision, recall, curve.total_gt, curve.num_predictions)

    heading_precision, heading_recall = _precision_recall(
        curve.cumulative_heading_tp, cumulative_fp, curve.total_gt
    )
    f1_scores = _f1_scores(precision, recall)
    optimal_index = _max_f1_index(f1_scores)

    if optimal_index >= 0:
        max_f1 = float(f1_scores[optimal_index])
        optimal_conf = float(curve.scores[optimal_index])
        optimal_recall = float(recall[optimal_index])
        optimal_precision = float(precision[optimal_index])
    else:
        max_f1 = optimal_conf = optimal_recall = optimal_precision = np.nan

    return CurveMetrics(
        ap=ap,
        aph=_interpolated_ap(
            heading_precision, heading_recall, curve.total_gt, curve.num_predictions
        ),
        max_f1=max_f1,
        optimal_conf=optimal_conf,
        optimal_index=optimal_index,
        optimal_recall=optimal_recall,
        optimal_precision=optimal_precision,
    )


def _precision_recall(
    cumulative_tp: np.ndarray,
    cumulative_fp: np.ndarray,
    total_gt: int,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = cumulative_tp + cumulative_fp
    precision = np.divide(
        cumulative_tp,
        denominator,
        out=np.zeros_like(cumulative_tp, dtype=np.float64),
        where=denominator != 0.0,
    )
    recall = cumulative_tp / float(total_gt) if total_gt > 0 else np.zeros_like(cumulative_tp)
    return precision, recall


def _interpolated_ap(
    precision: np.ndarray,
    recall: np.ndarray,
    total_gt: int,
    num_predictions: int,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> float:
    if total_gt == 0 and num_predictions == 0:
        return np.nan
    if precision.shape[0] == 0:
        return 0.0

    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, 101)
    precision_interp = np.interp(recall_grid, recall, precision_envelope, right=0.0)
    first_index = int(round(100 * min_recall)) + 1
    filtered_precision = precision_interp[first_index:] - min_precision
    filtered_precision[filtered_precision < 0.0] = 0.0
    return float(np.mean(filtered_precision)) / (1.0 - min_precision)


def _f1_scores(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denominator = precision + recall
    return np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator != 0.0,
    )


def _max_f1_index(f1_scores: np.ndarray) -> int:
    if f1_scores.shape[0] == 0 or np.all(np.isnan(f1_scores)):
        return -1
    return int(np.nanargmax(f1_scores))


def select_recall_tp_errors(curve: MatchCurve, recall_target: float) -> SelectedTpErrors:
    """Mean TP errors over the matches up to a recall target.

    Args:
        curve: Score-ordered match curve.
        recall_target: Recall level up to which matches are pooled.

    Returns:
        The mean TP errors over the selected matches.
    """
    effective_recall = (int(round(100 * recall_target)) + 1) / 100.0
    target_matches = int(np.floor(curve.total_gt * effective_recall))
    tp_indices = np.flatnonzero(curve.true_positive)[:target_matches]
    return _selected_error_values(curve, tp_indices)


def select_optimal_tp_errors(curve: MatchCurve, optimal_index: int) -> SelectedTpErrors:
    """Mean TP errors over the matches up to the optimal-F1 operating point.

    Args:
        curve: Score-ordered match curve.
        optimal_index: Index of the optimal-F1 operating point.

    Returns:
        The mean TP errors over the selected matches.
    """
    if optimal_index < 0:
        return _selected_error_values(curve, np.asarray([], dtype=np.int64))
    prefix_true_positive = curve.true_positive[: optimal_index + 1]
    tp_indices = np.flatnonzero(prefix_true_positive)
    return _selected_error_values(curve, tp_indices)


def _selected_error_values(curve: MatchCurve, tp_indices: np.ndarray) -> SelectedTpErrors:
    errors = {
        "ATE": _mean_or_one(curve.translation_error[tp_indices]),
        "AOE": _mean_or_one(curve.orientation_error[tp_indices]),
        "ASE": _mean_or_one(curve.scale_error[tp_indices]),
        "AVE": _mean_or_one(curve.velocity_error[tp_indices]),
        "AAE": _mean_or_one(curve.attribute_error[tp_indices]),
    }
    return SelectedTpErrors(count=int(tp_indices.shape[0]), errors=errors)


def mean_tp_errors(error_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each error name across the given per-class/threshold error dicts.

    Args:
        error_dicts: Error dicts collected per class and threshold.

    Returns:
        Mean value per error name.
    """
    return {
        error_name: mean_valid([errors[error_name] for errors in error_dicts])
        for error_name in ERROR_NAMES
    }


def nds(mean_ap: float, errors: dict[str, float]) -> float:
    """nuScenes detection score from mean AP and the mean TP errors.

    Args:
        mean_ap: Mean AP over classes and thresholds.
        errors: Mean TP errors by name.

    Returns:
        The detection score in ``[0, 1]``.
    """
    error_score = sum(max(0.0, 1.0 - errors[name]) for name in ERROR_NAMES)
    return (5.0 * mean_ap + error_score) / 10.0


def _orientation_errors(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Absolute yaw error wrapped to ``[0, pi]``, per matched pair."""
    diff = np.abs(pred_boxes[:, 6] - gt_boxes[:, 6])
    return np.abs((diff + pi) % (2.0 * pi) - pi)


def _heading_scores(orientation_errors: np.ndarray) -> np.ndarray:
    return np.round(np.clip(1.0 - orientation_errors / pi, 0.0, 1.0), 10)


def _scale_errors(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """``1 - IoU`` of the aligned (center- and yaw-matched) 3D dimensions."""
    pred_dims = np.maximum(pred_boxes[:, 3:6], 0.0)
    gt_dims = np.maximum(gt_boxes[:, 3:6], 0.0)
    intersection = np.prod(np.minimum(pred_dims, gt_dims), axis=1)
    union = np.prod(pred_dims, axis=1) + np.prod(gt_dims, axis=1) - intersection
    errors = np.ones(pred_boxes.shape[0], dtype=np.float64)
    positive = union > 0.0
    errors[positive] = 1.0 - intersection[positive] / union[positive]
    return errors


def _velocity_errors(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if pred_boxes.shape[1] < 9 or gt_boxes.shape[1] < 9:
        return np.ones(pred_boxes.shape[0], dtype=np.float64)
    return np.linalg.norm(pred_boxes[:, 7:9] - gt_boxes[:, 7:9], axis=1)


def _mean_or_one(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.shape[0] == 0:
        return 1.0
    return float(np.mean(valid))


def mean_valid(values: list[float] | tuple[float, ...]) -> float:
    """Mean of the non-NaN values, or NaN when none are valid.

    Args:
        values: Values that may contain NaN.

    Returns:
        The mean of the valid values.
    """
    valid_values = [float(value) for value in values if not np.isnan(float(value))]
    if not valid_values:
        return np.nan
    return float(sum(valid_values) / len(valid_values))


@dataclass
class DetectionState:
    """Synced detection state the suite hands to each metric.

    Attributes:
        samples: Per-frame samples (already GT-filtered, optionally range-clipped).
        class_names: Ordered class names for metric keys, or ``None`` to fall
            back to ``class_{label}`` tokens in every metric key.
        match_cost: Matching cost name (``center`` or ``corner``). Each metric asks
            for the match curves at whatever thresholds it needs.
    """

    samples: list[Detection3DSample]
    class_names: tuple[str, ...] | None
    match_cost: str = "center"
    _curve_cache: dict[tuple[int, float], MatchCurve] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # Prepared per-label boxes (masking, NumPy conversion, cost matrices) are
    # threshold-independent, so they are built once per label (PreparedLabel)
    # and every threshold's matching reuses them.
    _label_cache: dict[int, PreparedLabel] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _labels_cache: dict[bool, list[int]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _metrics_cache: dict[tuple[int, float], CurveMetrics] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def labels(self, full: bool) -> list[int]:
        """Class labels to report, memoized because every component asks.

        Args:
            full: Report every configured class when true, otherwise only
                classes that actually have ground truth.

        Returns:
            The class indices to report.
        """
        cached = self._labels_cache.get(bool(full))
        if cached is None:
            cached = labels_to_evaluate(self.samples, self.class_names if full else None)
            self._labels_cache[bool(full)] = cached
        return cached

    def curve_metrics(self, label: int, threshold: float) -> CurveMetrics:
        """AP/APH/F1 summary of a match curve, memoized like the curve itself.

        Several components read the same summary (mAP, APH, NDS, TP errors), so
        the cumulative passes over every prediction of the class run once here.

        Args:
            label: Class label.
            threshold: Match threshold in meters.

        Returns:
            The memoized curve summary.
        """
        key = (label, float(threshold))
        metrics = self._metrics_cache.get(key)
        if metrics is None:
            metrics = curve_metrics(self.match_curve(label, threshold))
            self._metrics_cache[key] = metrics
        return metrics

    def match_curve(self, label: int, threshold: float) -> MatchCurve:
        """Return the score-ordered match curve for a class and threshold.

        Memoized, so the matching runs once per ``(label, threshold)`` and is
        shared by every metric that asks for it.

        Args:
            label: Class label.
            threshold: Match threshold in meters.

        Returns:
            The memoized match curve.
        """
        key = (label, float(threshold))
        curve = self._curve_cache.get(key)
        if curve is None:
            prepared = self._label_cache.get(label)
            if prepared is None:
                prepared = PreparedLabel(self.samples, label, resolve_match_cost(self.match_cost))
                self._label_cache[label] = prepared
            curve = prepared.match(threshold)
            self._curve_cache[key] = curve
        return curve
