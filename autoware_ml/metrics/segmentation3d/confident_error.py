"""Confident-error rate.

An error the model is unsure about is recoverable, a confident error is what
triggers a hard brake or hides an obstacle. Over the misclassified points
(prediction disagrees with ground truth, no class partitioning), this reports the
fraction made at high confidence, i.e. low predictive entropy. Like the error-cluster rate, the area
comes from an attached lanelet ``RegionFilter``: on the road this is the
confident-phantom rate, whole-scene it characterizes the model overall.
"""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.point_cloud import (
    PointCloudSegState,
    valid_point_mask,
)


class ConfidentErrorRate(Metric[PointCloudSegState]):
    """Fraction of misclassified points made at high confidence (low entropy)."""

    def __init__(
        self,
        entropy_threshold: float = 0.3,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Store the confidence boundary.

        Args:
            entropy_threshold: Normalized-entropy value below which an error counts as
                confident.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.entropy_threshold = float(entropy_threshold)

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Split misclassified points by confidence across frames.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        error_total = confident_total = 0
        for frame in state.frames:
            valid = valid_point_mask(frame, state.num_classes, state.ignore_index)
            if not valid.any():
                continue
            wrong = frame.pred[valid] != frame.target[valid]
            if not wrong.any():
                continue
            entropy = frame.entropy[valid][wrong]
            error_total += int(wrong.sum())
            confident_total += int((entropy < self.entropy_threshold).sum())

        return {
            "confident_error_rate": (
                (confident_total / error_total) if error_total else float("nan")
            ),
            "confident_error_count": float(confident_total),
        }
