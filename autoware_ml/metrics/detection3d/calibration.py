"""Detection calibration error.

A detection score should match the empirical precision: of the boxes scored about
0.9, roughly 90% should be true positives. This bins the predictions by score and
measures the gap between the mean score and the precision in each bin, mirroring
the segmentation calibration metric on the per-point softmax. It reads the
score-ordered true/false-positive flags already on the match curve, so it adds no
state. Measured as shipped, before any post-hoc score recalibration.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import DetectionState, mean_valid
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


def _expected_calibration_error(scores: np.ndarray, correct: np.ndarray, num_bins: int) -> float:
    """Prediction-weighted mean gap between confidence and correctness over score bins."""
    if scores.shape[0] == 0:
        return float("nan")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError(
            "detection scores must be probabilities in [0, 1] but values fall outside, "
            "calibrate against probabilities, not raw logits."
        )
    total = scores.shape[0]
    bin_index = np.minimum((scores * num_bins).astype(np.int64), num_bins - 1)
    count = np.bincount(bin_index, minlength=num_bins).astype(np.float64)
    score_sum = np.bincount(bin_index, weights=scores, minlength=num_bins)
    correct_sum = np.bincount(bin_index, weights=correct, minlength=num_bins)
    nonempty = count > 0
    gaps = np.abs(correct_sum[nonempty] / count[nonempty] - score_sum[nonempty] / count[nonempty])
    return float(np.sum(count[nonempty] / total * gaps))


class CalibrationError(Metric[DetectionState]):
    """Expected calibration error of the detection score against precision.

    ``ece`` pools every prediction and ``ece_macro`` averages the per-class ECE so a
    dominant class does not hide a badly calibrated rare one. A prediction is
    "correct" when it is a true positive at ``tp_threshold``, so a bin's accuracy is
    its precision.
    """

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        num_bins: int = 15,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the bin count.

        Args:
            tp_threshold: Center-distance match threshold in meters deciding correctness.
            num_bins: Number of equal-width score bins.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.tp_threshold = float(tp_threshold)
        self.num_bins = int(num_bins)
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1.")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Report overall and macro calibration error at the TP threshold.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        all_scores: list[np.ndarray] = []
        all_correct: list[np.ndarray] = []
        per_class: list[float] = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            if curve.scores.shape[0] == 0:
                continue
            all_scores.append(curve.scores)
            all_correct.append(curve.true_positive)
            per_class.append(
                _expected_calibration_error(curve.scores, curve.true_positive, self.num_bins)
            )
        if not all_scores:
            return {"ece": float("nan"), "ece_macro": float("nan")}
        scores = np.concatenate(all_scores)
        correct = np.concatenate(all_correct)
        return {
            "ece": _expected_calibration_error(scores, correct, self.num_bins),
            "ece_macro": mean_valid(per_class),
        }
