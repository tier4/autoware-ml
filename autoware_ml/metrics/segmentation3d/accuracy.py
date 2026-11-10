"""Overall accuracy metric."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.report import MetricReport
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class Accuracy(Metric[ConfusionState]):
    """Micro point accuracy: correct points over all valid points. A single global
    ratio, so there is no per-class breakdown.
    """

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> MetricReport:
        """Compute global point accuracy from a confusion state.

        Args:
            state: Segmentation confusion state.
            stage: Evaluation stage requesting the metric.

        Returns:
            The accuracy.
        """
        report = MetricReport()
        if state.total > 0:
            report.record("acc", float((state.true_positive.sum() / state.total).item()))
        else:
            report.record("acc", float("nan"))
        return report
