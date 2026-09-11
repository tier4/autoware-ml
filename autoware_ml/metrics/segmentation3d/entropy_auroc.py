"""Uncertainty usefulness, entropy as an error detector.

Uncertainty is only worth anything if it flags the model's own mistakes. This
measures how well per-point predictive entropy separates correct from
misclassified points, via the AUROC of entropy for the binary "is this point
wrong?" task. 0.5 means the uncertainty is meaningless, higher means it is a
usable error flag. Diagnostic only: it characterizes the uncertainty signal, it
does not rank accuracy.

Entropy is normalized to ``[0, 1]``, so the AUROC is computed from two
fixed-width histograms (wrong / correct) accumulated per frame. Memory stays
O(num_bins) instead of pooling every point of the test split, and the tie-aware
closed form over the bins quantizes entropy to ``1 / num_bins`` (an error of at
most one bin width in the separation, negligible at the default resolution).
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.point_cloud import (
    PointCloudSegState,
    valid_point_mask,
)


def binned_auroc(positive_hist: np.ndarray, negative_hist: np.ndarray) -> float:
    """Tie-aware AUROC of a score for the positive class, from two histograms.

    Treats all values inside one bin as tied (the standard mid-rank convention):
    ``P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)``.

    Args:
        positive_hist: Score histogram of the positive class.
        negative_hist: Score histogram of the negative class.

    Returns:
        The AUROC estimate in ``[0, 1]``.
    """
    num_positive = float(positive_hist.sum())
    num_negative = float(negative_hist.sum())
    if num_positive == 0.0 or num_negative == 0.0:
        return float("nan")
    negative_below = np.concatenate(([0.0], np.cumsum(negative_hist)[:-1]))
    wins = float(np.sum(positive_hist * negative_below))
    ties = float(np.sum(positive_hist * negative_hist))
    return (wins + 0.5 * ties) / (num_positive * num_negative)


class UncertaintyUsefulness(Metric[PointCloudSegState]):
    """AUROC of entropy as a misclassification detector, plus mean entropy split."""

    def __init__(
        self,
        num_bins: int = 8192,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the histogram resolution.

        Args:
            num_bins: Number of fixed-width entropy bins the AUROC is computed over.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.num_bins = int(num_bins)
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2.")

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Accumulate per-frame entropy histograms and score the AUROC.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        wrong_hist = np.zeros(self.num_bins, dtype=np.float64)
        correct_hist = np.zeros(self.num_bins, dtype=np.float64)
        wrong_entropy = correct_entropy = 0.0
        wrong_count = correct_count = 0
        for frame in state.frames:
            valid = valid_point_mask(frame, state.num_classes, state.ignore_index)
            if not valid.any():
                continue
            entropy = frame.entropy[valid]
            wrong = frame.pred[valid] != frame.target[valid]
            bins = np.minimum((entropy * self.num_bins).astype(np.int64), self.num_bins - 1)
            wrong_hist += np.bincount(bins[wrong], minlength=self.num_bins)
            correct_hist += np.bincount(bins[~wrong], minlength=self.num_bins)
            wrong_entropy += float(entropy[wrong].sum(dtype=np.float64))
            correct_entropy += float(entropy[~wrong].sum(dtype=np.float64))
            wrong_count += int(wrong.sum())
            correct_count += int((~wrong).sum())

        # Every key is always present, a bucket without errors reports NaN rather
        # than a key that comes and goes between epochs.
        return {
            "entropy_auroc": binned_auroc(wrong_hist, correct_hist),
            "mean_entropy_wrong": wrong_entropy / wrong_count if wrong_count else float("nan"),
            "mean_entropy_correct": (
                correct_entropy / correct_count if correct_count else float("nan")
            ),
        }
