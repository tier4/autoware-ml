"""Number of evaluated frames."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import Detection3DState
from autoware_ml.metrics.report import MetricReport


class FrameCount(Metric[Detection3DState]):
    """Number of frames the other metrics scored.

    The count is taken from the state the suite hands over, so a region-filtered
    component reports only the frames its filter covered (a scene without a
    lanelet map is omitted from that slice). Range clipping removes boxes, never
    frames, so the count is range-invariant and reported once per slice: an
    empty frame still counts, it is a frame the model was evaluated on.
    """

    range_invariant = True

    def evaluate(self, state: Detection3DState, stage: EvalStage) -> MetricReport:
        """Count the accumulated frames.

        Args:
            state: Detection state holding the per-frame samples.
            stage: Evaluation stage requesting the metric.

        Returns:
            The frame count.
        """
        report = MetricReport()
        report.record("num_frames", float(len(state.samples)))
        return report
