"""Detection confident-error rate.

Of the boxes the model got wrong, what fraction was it sure about? A high-score
false positive is the confident phantom that triggers a hard brake, so it is the
release blocker. This is false-positive only by nature: a false negative carries no
prediction and therefore no score, so the missed-object (safety) axis belongs to the
critical false-negative metric. It reads the score-ordered false-positive flags already on the match
curve and adds no new state.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import DetectionState
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class ConfidentErrorRate(Metric[DetectionState]):
    """Confident-FP rate: of the false positives above ``min_score`` (the
    reporting floor), the fraction scored at or above ``score_threshold``."""

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        score_threshold: float = 0.5,
        min_score: float = 0.1,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the score floor and the confident threshold.

        Args:
            tp_threshold: Center-distance match threshold in meters deciding false positives.
            score_threshold: Score at or above which a false positive counts as confident.
            min_score: Reporting floor, false positives below it leave the denominator.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.tp_threshold = float(tp_threshold)
        self.score_threshold = float(score_threshold)
        # Denominator floor: detection heads emit a long near-zero-score FP tail
        # no consumer ever sees. Without a floor the rate tracks the head's score
        # cutoff more than its confident-error behaviour.
        self.min_score = float(min_score)
        if not 0.0 <= self.min_score <= self.score_threshold <= 1.0:
            raise ValueError("expected 0 <= min_score <= score_threshold <= 1.")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Report the confident false-positive rate over all classes.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        false_positive_total = 0
        confident_false_positive_total = 0
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            false_positive_scores = curve.scores[curve.false_positive]
            false_positive_total += int(np.sum(false_positive_scores >= self.min_score))
            confident_false_positive_total += int(
                np.sum(false_positive_scores >= self.score_threshold)
            )
        rate = (
            confident_false_positive_total / false_positive_total
            if false_positive_total
            else float("nan")
        )
        num_frames = len(state.samples) if state.samples else float("nan")
        return {
            "confident_error_rate": rate,
            "confident_error_count": float(confident_false_positive_total),
            "confident_errors_per_frame": confident_false_positive_total / num_frames,
        }
