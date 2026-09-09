"""Detection confusion matrix (matched detections only).

Class-agnostic greedy center-distance matching pairs each prediction to its
nearest unclaimed ground-truth box within ``match_threshold``, and every matched
pair adds one to ``confusion[true_class, pred_class]``. Predictions below
``min_score`` and unmatched boxes are dropped: this is the label-confusion view
among detections that did match (a car predicted where a truck stands lands off
the diagonal), not a recall matrix. In a grouped suite the state's labels are
already folded, so the matrix is over behaviour groups with no change here.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter
from autoware_ml.metrics.confusion_report import confusion_cells
from autoware_ml.metrics.detection3d.criticality import greedy_match
from autoware_ml.metrics.detection3d.matching import DetectionState
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class ConfusionMatrix(Metric[DetectionState]):
    """True vs. predicted class counts over class-agnostically matched detections."""

    def __init__(
        self,
        match_threshold: float = DEFAULT_TP_THRESHOLD,
        min_score: float = 0.1,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter: MetricFilter | None = None,
    ) -> None:
        """Validate the score floor.

        Args:
            match_threshold: Class-agnostic center-distance match threshold in meters.
            min_score: Reporting floor, predictions below it never enter the matrix.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.match_threshold = float(match_threshold)
        self.min_score = float(min_score)
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1].")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Accumulate matched (true, predicted) label pairs into a confusion matrix.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        if state.class_names is None:
            raise ValueError("ConfusionMatrix needs class_names to label its cells.")
        if state.match_cost != "center":
            raise ValueError(
                "ConfusionMatrix matches class-agnostically by center distance, a suite "
                f"configured with match_cost={state.match_cost!r} would report matched "
                "pairs inconsistent with its AP family."
            )
        num_classes = len(state.class_names)
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for sample in state.samples:
            if sample.gt_boxes.shape[0] == 0 or sample.pred_boxes.shape[0] == 0:
                continue
            gt_centers = sample.gt_boxes[:, :2].cpu().numpy()
            pred_scores = sample.pred_scores.cpu().numpy()
            # The near-zero-score FP tail no consumer ever sees must not fill the
            # off-diagonal cells, only predictions above the reporting floor match.
            keep = pred_scores >= self.min_score
            pred_centers = sample.pred_boxes[:, :2].cpu().numpy()[keep]
            if pred_centers.shape[0] == 0:
                continue
            is_tp, matched_gt = greedy_match(
                gt_centers, pred_centers, pred_scores[keep], self.match_threshold
            )
            true_labels = sample.gt_labels.cpu().numpy()[matched_gt[is_tp]]
            pred_labels = sample.pred_labels.cpu().numpy()[keep][is_tp]
            pairs = np.concatenate([true_labels, pred_labels])
            if pairs.shape[0] and (pairs.min() < 0 or pairs.max() >= num_classes):
                raise ValueError(
                    "matched labels outside the configured class range, a folding or "
                    "configuration bug upstream."
                )
            np.add.at(matrix, (true_labels, pred_labels), 1)
        return confusion_cells(matrix, state.class_names)
