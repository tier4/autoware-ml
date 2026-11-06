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

"""Shared frame-local matching and weighted-AP helpers for the detection metrics.

The reachability time-to-collision and its risk weight live in
:mod:`autoware_ml.metrics.geometry.reachability`, and the criticality metrics read a per-box TTC the suite has already computed. This module holds only the
stateless NumPy pieces several metrics share, the greedy center-distance match
and the weighted average precision, so they are tested without the suite.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float, Int64


class GreedyMatch(NamedTuple):
    """Result of one greedy claim pass, aligned to the predictions in their original order.

    Attributes:
        is_tp: Whether the prediction claimed a ground truth box the metrics count.
        matched_gt: Index of the claimed ground truth box, -1 for an unmatched prediction.
        ignored: Whether the prediction claimed an ignored ground truth box instead, which
            leaves it out of the evaluation as neither a true nor a false positive.
    """

    is_tp: Bool[np.ndarray, " num_preds"]
    matched_gt: Int64[np.ndarray, " num_preds"]
    ignored: Bool[np.ndarray, " num_preds"]


def greedy_match_costs(
    costs: Float[np.ndarray, "num_preds num_gts"],
    ignored_costs: Float[np.ndarray, "num_preds num_ignored"],
    pred_scores: Float[np.ndarray, " num_preds"],
    thresholds: tuple[float, ...],
) -> dict[float, GreedyMatch]:
    """Score-ordered greedy matching on a cost matrix at several thresholds.

    In score order every prediction claims its nearest unclaimed counted box within the
    threshold. A prediction left without one claims its nearest unclaimed ignored box within
    the threshold instead, one prediction per box, so a duplicate stays a false positive. The
    candidate order and the score order are threshold independent, so they are built once and
    each threshold runs only its own claim pass.

    Args:
        costs: Match cost of every prediction and counted ground truth pair ``(N, M)``.
        ignored_costs: Match cost of every prediction and ignored ground truth pair ``(N, I)``.
        pred_scores: Prediction confidences, shape ``(N,)``.
        thresholds: Largest match cost of each claim pass.

    Returns:
        Mapping of threshold to the match of its claim pass.
    """
    num_pred = costs.shape[0]
    matches = {
        threshold: GreedyMatch(
            np.zeros(num_pred, dtype=bool),
            np.full(num_pred, -1, dtype=np.int64),
            np.zeros(num_pred, dtype=bool),
        )
        for threshold in thresholds
    }
    if num_pred == 0:
        return matches
    # Candidates in ascending cost order, computed once: each claim pass then only walks a row
    # past its claimed entries. The stable sort keeps the lowest index first among equal
    # costs, the argmin tie rule.
    candidates = np.argsort(costs, axis=1, kind="stable").tolist()
    ignored_candidates = np.argsort(ignored_costs, axis=1, kind="stable").tolist()
    score_order = np.argsort(-pred_scores, kind="stable")
    for threshold, match in matches.items():
        claimed = np.zeros(costs.shape[1], dtype=bool)
        claimed_ignored = np.zeros(ignored_costs.shape[1], dtype=bool)
        for pred_index in score_order:
            candidate = _first_unclaimed(
                costs[pred_index], candidates[pred_index], claimed, threshold
            )
            if candidate >= 0:
                match.is_tp[pred_index] = True
                match.matched_gt[pred_index] = candidate
                continue
            candidate = _first_unclaimed(
                ignored_costs[pred_index],
                ignored_candidates[pred_index],
                claimed_ignored,
                threshold,
            )
            match.ignored[pred_index] = candidate >= 0
    return matches


def _first_unclaimed(
    row: Float[np.ndarray, " num_gts"],
    candidates: list[int],
    claimed: Bool[np.ndarray, " num_gts"],
    threshold: float,
) -> int:
    """Claim the nearest unclaimed box within the threshold and return it, -1 for none.

    Args:
        row: Match cost of the prediction to every box.
        candidates: Box indices in ascending cost order.
        claimed: Claimed flags of the boxes, updated in place.
        threshold: Largest match cost.

    Returns:
        Index of the claimed box, -1 when no unclaimed box lies within the threshold.
    """
    for candidate in candidates:
        if row[candidate] > threshold:
            return -1
        if not claimed[candidate]:
            claimed[candidate] = True
            return candidate
    return -1


def center_distances(
    pred_centers: Float[np.ndarray, "num_preds _"], gt_centers: Float[np.ndarray, "num_gts _"]
) -> Float[np.ndarray, "num_preds num_gts"]:
    """BEV center distance of every prediction and ground truth pair."""
    return np.linalg.norm(pred_centers[:, None, :2] - gt_centers[None, :, :2], axis=2)


def greedy_match_thresholds(
    gt_centers: Float[np.ndarray, "num_gts _"],
    ignored_centers: Float[np.ndarray, "num_ignored _"],
    pred_centers: Float[np.ndarray, "num_preds _"],
    pred_scores: Float[np.ndarray, " num_preds"],
    thresholds: tuple[float, ...],
) -> dict[float, GreedyMatch]:
    """Score-ordered greedy center-distance matching at several thresholds.

    Args:
        gt_centers: Counted ground truth BEV centers, shape ``(M, 2+)``.
        ignored_centers: Ignored ground truth BEV centers, shape ``(I, 2+)``.
        pred_centers: Predicted BEV centers, shape ``(N, 2+)``.
        pred_scores: Prediction confidences, shape ``(N,)``.
        thresholds: Center-distance match thresholds in meters.

    Returns:
        Mapping of threshold to the match of its claim pass, see :func:`greedy_match_costs`.
    """
    return greedy_match_costs(
        center_distances(pred_centers, gt_centers),
        center_distances(pred_centers, ignored_centers),
        pred_scores,
        thresholds,
    )


def greedy_match(
    gt_centers: Float[np.ndarray, "num_gts _"],
    ignored_centers: Float[np.ndarray, "num_ignored _"],
    pred_centers: Float[np.ndarray, "num_preds _"],
    pred_scores: Float[np.ndarray, " num_preds"],
    threshold: float,
) -> GreedyMatch:
    """Score-ordered greedy center-distance matching for one frame and class.

    Args:
        gt_centers: Counted ground truth BEV centers ``(G, 2+)``.
        ignored_centers: Ignored ground truth BEV centers ``(I, 2+)``.
        pred_centers: Prediction BEV centers ``(P, 2+)``.
        pred_scores: Prediction confidences ``(P,)``.
        threshold: Maximum center distance for a match, in meters.

    Returns:
        The match of the claim pass, aligned to the predictions in their original order.
    """
    return greedy_match_thresholds(
        gt_centers, ignored_centers, pred_centers, pred_scores, (threshold,)
    )[threshold]


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
