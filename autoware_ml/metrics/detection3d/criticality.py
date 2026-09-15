"""Shared frame-local matching and weighted-AP helpers for the detection metrics.

The reachability time-to-collision and its risk weight live in
:mod:`autoware_ml.metrics.geometry.reachability`, and the criticality metrics read a per-box TTC the suite has already computed. This module holds only the
stateless NumPy pieces several metrics share, the greedy center-distance match
and the weighted average precision, so they are tested without the suite.
"""

from __future__ import annotations

import numpy as np


def greedy_match_thresholds(
    gt_centers: np.ndarray,
    pred_centers: np.ndarray,
    pred_scores: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Score-ordered greedy center-distance matching at several thresholds.

    The distance matrix, the candidate order, and the score order are threshold
    independent, so they are built once and each threshold runs only its own
    claim pass.

    Args:
        gt_centers: Ground-truth BEV centers, shape ``(M, 2+)``.
        pred_centers: Predicted BEV centers, shape ``(N, 2+)``.
        pred_scores: Prediction confidences, shape ``(N,)``.
        thresholds: Center-distance match thresholds in meters.

    Returns:
        Mapping of threshold to ``(is_tp, matched_gt)`` arrays aligned to the
        predictions in their original order. ``is_tp`` is a boolean mask and
        ``matched_gt`` holds the matched ground-truth index, ``-1`` for
        non-matches.
    """
    num_pred = pred_centers.shape[0]
    empty = {
        threshold: (np.zeros(num_pred, dtype=bool), np.full(num_pred, -1, dtype=np.int64))
        for threshold in thresholds
    }
    if num_pred == 0 or gt_centers.shape[0] == 0:
        return empty
    distances = np.linalg.norm(pred_centers[:, None, :2] - gt_centers[None, :, :2], axis=2)
    # Candidates in ascending distance order, computed once: each claim pass
    # then only walks a row past its claimed entries. The stable sort keeps
    # the lowest index first among equal distances, the argmin tie rule.
    candidates = np.argsort(distances, axis=1, kind="stable").tolist()
    score_order = np.argsort(-pred_scores, kind="stable")
    for threshold, (is_tp, matched_gt) in empty.items():
        claimed = np.zeros(gt_centers.shape[0], dtype=bool)
        for pred_index in score_order:
            row = distances[pred_index]
            for candidate in candidates[pred_index]:
                if row[candidate] > threshold:
                    break
                if claimed[candidate]:
                    continue
                is_tp[pred_index] = True
                matched_gt[pred_index] = candidate
                claimed[candidate] = True
                break
    return empty


def greedy_match(
    gt_centers: np.ndarray, pred_centers: np.ndarray, pred_scores: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Score-ordered greedy center-distance matching for one frame and class.

    Args:
        gt_centers: Ground-truth BEV centers ``(G, 2)``.
        pred_centers: Prediction BEV centers ``(P, 2)``.
        pred_scores: Prediction confidences ``(P,)``.
        threshold: Maximum center distance for a match, in meters.

    Returns:
        ``(is_tp, matched_gt)`` aligned to the predictions in their original
        order, with the matched GT index ``-1`` for unmatched predictions.
    """
    return greedy_match_thresholds(gt_centers, pred_centers, pred_scores, (threshold,))[threshold]


def weighted_average_precision(
    weights: np.ndarray,
    is_tp: np.ndarray,
    scores: np.ndarray,
    total_gt_weight: float,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> float:
    """nuScenes-style interpolated AP with per-prediction weights.

    ``weights`` weighs each prediction's TP/FP contribution and ``total_gt_weight``
    is the summed weight of all ground truth (the recall denominator).

    Args:
        weights: Per-prediction weights, aligned with ``scores``.
        is_tp: Per-prediction TP flags.
        scores: Prediction confidences.
        total_gt_weight: Summed weight of all ground truth.
        min_recall: Recall region excluded from the AP integral.
        min_precision: Precision baseline subtracted before normalization.

    Returns:
        Normalized interpolated AP in ``[0, 1]``.
    """
    if total_gt_weight <= 0.0:
        return float("nan")
    if scores.shape[0] == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_weight = np.where(is_tp[order], weights[order], 0.0)
    fp_weight = np.where(is_tp[order], 0.0, weights[order])
    cum_tp = np.cumsum(tp_weight)
    cum_fp = np.cumsum(fp_weight)
    denominator = cum_tp + cum_fp
    precision = np.divide(cum_tp, denominator, out=np.zeros_like(cum_tp), where=denominator != 0.0)
    recall = cum_tp / total_gt_weight
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = np.interp(recall_grid, recall, envelope, right=0.0)
    first = int(round(100 * min_recall)) + 1
    filtered = interpolated[first:] - min_precision
    filtered[filtered < 0.0] = 0.0
    return float(np.mean(filtered)) / (1.0 - min_precision)
