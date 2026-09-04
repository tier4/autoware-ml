"""Collision-risk-weighted mAP.

Every object's contribution to precision and recall is weighted by ``w =
e^(-decay * TTC)``, with TTC the reachability time-to-collision to ego computed
once per frame by the suite's collision provider: each agent and ego are
propagated at their class max speed, and TTC is the first time their reachable
sets overlap. An object that cannot be reached within the horizon has TTC = inf
and weight 0, so a same-speed lead or off-road scenery buys no score,
while an imminent in-path object dominates it.

Matching is the usual per-frame greedy center distance, only the precision and
recall accumulation is weighted. Reported alongside the unweighted mAP, never
instead.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.criticality import (
    greedy_match_thresholds,
    weighted_average_precision,
)
from autoware_ml.metrics.detection3d.matching import DetectionState, mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.structures import DEFAULT_MATCH_THRESHOLDS
from autoware_ml.metrics.geometry.reachability import collision_weights


class CollisionWeightedMeanAP(Metric[DetectionState]):
    """Class-mean AP with each object weighted by its reachability collision risk."""

    needs_ttc = True

    def __init__(
        self,
        thresholds: tuple[float, ...] = DEFAULT_MATCH_THRESHOLDS,
        decay: float = 0.5,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the risk decay.

        Args:
            thresholds: Center-distance match thresholds in meters, the weighted AP is averaged
                over them.
            decay: Exponential risk decay per second of TTC, must be non-negative.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        self.decay = float(decay)
        if self.decay < 0.0:
            raise ValueError("decay must be >= 0.")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Weighted AP per class, averaged over thresholds, then over classes.

        One pass over the frames: tensors are converted and TTC-weighted once per
        frame, and the greedy matching shares its cost matrix across thresholds.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        labels = state.labels(full=True)
        total_gt_weight = {label: 0.0 for label in labels}
        scores: dict[tuple[int, float], list[np.ndarray]] = {
            (label, threshold): [] for label in labels for threshold in self.thresholds
        }
        weights = {key: [] for key in scores}
        is_tp = {key: [] for key in scores}
        for sample in state.samples:
            if sample.gt_ttc is None or sample.pred_ttc is None:
                raise ValueError(
                    "CollisionWeightedMeanAP needs per-box TTC, build the suite with a "
                    "collision provider."
                )
            # A frame without a lanelet map has no TTC basis: excluded from the
            # weighted mass entirely, mirroring the region-filter coverage rule.
            if not sample.ttc_covered:
                continue
            gt_labels = sample.gt_labels.numpy()
            gt_centers = sample.gt_boxes.numpy().astype(np.float64)[:, :2]
            gt_weight_all = collision_weights(sample.gt_ttc.numpy(), self.decay)
            pred_labels = sample.pred_labels.numpy()
            pred_centers = sample.pred_boxes.numpy().astype(np.float64)[:, :2]
            pred_scores_all = sample.pred_scores.numpy().astype(np.float64)
            pred_weight_all = collision_weights(sample.pred_ttc.numpy(), self.decay)
            for label in labels:
                gt_mask = gt_labels == label
                gt_weight = gt_weight_all[gt_mask]
                total_gt_weight[label] += float(gt_weight.sum())

                pred_mask = pred_labels == label
                pred_scores = pred_scores_all[pred_mask]
                if pred_scores.shape[0] == 0:
                    continue
                matches = greedy_match_thresholds(
                    gt_centers[gt_mask], pred_centers[pred_mask], pred_scores, self.thresholds
                )
                for threshold, (tp, matched) in matches.items():
                    # A true positive inherits its matched GT's weight, a false
                    # positive keeps its own.
                    weight = pred_weight_all[pred_mask].copy()
                    weight[tp] = gt_weight[matched[tp]]
                    scores[(label, threshold)].append(pred_scores)
                    weights[(label, threshold)].append(weight)
                    is_tp[(label, threshold)].append(tp)

        per_class: dict[int, float] = {}
        for label in labels:
            per_class[label] = mean_valid(
                [
                    self._assemble_ap(
                        weights[(label, threshold)],
                        is_tp[(label, threshold)],
                        scores[(label, threshold)],
                        total_gt_weight[label],
                    )
                    for threshold in self.thresholds
                ]
            )
        report = {"cw_mAP": mean_valid(list(per_class.values()))}
        if stage is EvalStage.TEST:
            for label, value in per_class.items():
                report[f"cw_mAP_{label_metric_name(label, state.class_names)}"] = value
        return report

    @staticmethod
    def _assemble_ap(
        weights: list[np.ndarray],
        is_tp: list[np.ndarray],
        scores: list[np.ndarray],
        total_gt_weight: float,
    ) -> float:
        if not scores:
            return weighted_average_precision(
                np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0), total_gt_weight
            )
        return weighted_average_precision(
            np.concatenate(weights),
            np.concatenate(is_tp),
            np.concatenate(scores),
            total_gt_weight,
        )
