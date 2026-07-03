"""Overall accuracy metric."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class Accuracy(Metric[ConfusionState]):
    """Micro point accuracy: correct points over all valid points. A single global
    ratio, so there is no per-class breakdown.
    """

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> dict[str, float]:
        """Compute global point accuracy from a confusion state.

        Args:
            state: Segmentation confusion state.
            stage: Evaluation stage requesting the metric.

        Returns:
            Mapping containing the scalar accuracy value.
        """
        if state.total > 0:
            return {"acc": float((state.true_positive.sum() / state.total).item())}
        return {"acc": float("nan")}
