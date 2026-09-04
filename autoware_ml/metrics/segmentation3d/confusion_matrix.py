"""Segmentation confusion matrix (point level).

The confusion suite already accumulates the point confusion matrix (rows = true
class, columns = predicted class) as its state, folded onto behaviour groups in a
grouped suite, so this metric is a thin reader that emits its cells as
``confusion_<true>__<pred>`` counts.
"""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.confusion_report import confusion_cells
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class ConfusionMatrix(Metric[ConfusionState]):
    """Point true vs. predicted class counts, read straight from the suite's matrix."""

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> dict[str, float]:
        """Emit the accumulated confusion matrix as flat count keys.

        Args:
            state: Confusion state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        return confusion_cells(state.confusion.cpu().numpy(), state.class_names)
