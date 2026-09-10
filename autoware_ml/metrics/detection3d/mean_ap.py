"""Mean average precision metric."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import DetectionState, mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name, threshold_token
from autoware_ml.metrics.detection3d.structures import DEFAULT_MATCH_THRESHOLDS


class MeanAP(Metric[DetectionState]):
    """Class-mean AP. Validation reports only mAP and per-class AP, so epochs
    stay fast. Test adds the per-class GT count and the per-threshold AP curve
    details (match count, max F1, optimal-confidence operating point). The AP is
    averaged over ``thresholds`` (the nuScenes center-distance set by default).
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = DEFAULT_MATCH_THRESHOLDS,
        stages: tuple[str, ...] | list[str] = ("val", "test"),
        filter=None,
    ) -> None:
        """Validate the match thresholds.

        Args:
            thresholds: Center-distance match thresholds in meters, the AP is averaged over them.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        if not self.thresholds:
            raise ValueError("thresholds must not be empty.")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Compute mean AP metrics for the accumulated detection state.

        Args:
            state: Detection state with cached match curves.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of metric names to scalar values.
        """
        # Both stages average over the full trained class set. The val number
        # used for model selection and the reported test number therefore cover
        # the same classes, and the monitored key always exists, reporting NaN
        # when no class is measurable.
        labels = state.labels(full=True)
        if not labels:
            return {"mAP": float("nan")}

        per_class_ap = {
            label: mean_valid([state.curve_metrics(label, t).ap for t in self.thresholds])
            for label in labels
        }
        report = {"mAP": mean_valid(list(per_class_ap.values()))}
        for label, ap in per_class_ap.items():
            report[f"mAP_{label_metric_name(label, state.class_names)}"] = ap
        if stage is EvalStage.VAL:
            return report

        for label in labels:
            name = label_metric_name(label, state.class_names)
            report[f"gt_count_{name}"] = float(
                state.match_curve(label, self.thresholds[0]).total_gt
            )
            for threshold in self.thresholds:
                curve = state.match_curve(label, threshold)
                metrics = state.curve_metrics(label, threshold)
                token = threshold_token(threshold)
                report[f"AP_{name}_{token}"] = metrics.ap
                report[f"num_match_{name}_{token}"] = float(curve.num_match)
                report[f"max_f1_{name}_{token}"] = metrics.max_f1
                report[f"optimal_conf_{name}_{token}"] = metrics.optimal_conf
                report[f"optimal_recall_{name}_{token}"] = metrics.optimal_recall
                report[f"optimal_precision_{name}_{token}"] = metrics.optimal_precision
        return report
