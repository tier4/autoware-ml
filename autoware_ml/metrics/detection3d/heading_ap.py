"""Heading-aware average precision metric (test only by default)."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import curve_metrics, mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name, threshold_token
from autoware_ml.metrics.detection3d.structures import DetectionState


class HeadingAP(Metric[DetectionState]):
    """Class-mean APH plus per-class and per-threshold APH. APH weights each true
    positive by its heading score, so it needs the orientation errors the match
    curve carries.
    """

    def __init__(self, stages: tuple[str, ...] | list[str] = ("test",)) -> None:
        super().__init__(stages)

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Compute heading-aware AP metrics for the accumulated detection state.

        Args:
            state: Detection state with cached match curves.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of metric names to scalar values.
        """
        labels = state.labels(full=True)
        per_class_aph = {
            label: mean_valid(
                [curve_metrics(state.match_curve(label, t)).aph for t in state.thresholds]
            )
            for label in labels
        }
        report = {"mAPH": mean_valid(list(per_class_aph.values()))}
        for label, aph in per_class_aph.items():
            name = label_metric_name(label, state.class_names)
            report[f"mAPH_{name}"] = aph
            for threshold in state.thresholds:
                token = threshold_token(threshold)
                report[f"APH_{name}_{token}"] = curve_metrics(
                    state.match_curve(label, threshold)
                ).aph
        return report
