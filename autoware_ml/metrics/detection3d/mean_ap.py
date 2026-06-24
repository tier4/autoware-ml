"""Mean average precision metric."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import curve_metrics, mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name, threshold_token
from autoware_ml.metrics.detection3d.structures import DetectionState


class MeanAP(Metric[DetectionState]):
    """Class-mean AP. Validation reports only mAP and per-class AP, so epochs
    stay fast. Test adds the per-class GT count and the per-threshold AP curve
    details (match count, max F1, optimal-confidence operating point).
    """

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        full = stage is EvalStage.TEST
        labels = state.labels(full)
        if not labels:
            return {} if stage is EvalStage.VAL else {"mAP": float("nan")}

        per_class_ap = {
            label: mean_valid(
                [curve_metrics(state.match_curve(label, t)).ap for t in state.thresholds]
            )
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
                state.match_curve(label, state.thresholds[0]).total_gt
            )
            for threshold in state.thresholds:
                curve = state.match_curve(label, threshold)
                metrics = curve_metrics(curve)
                token = threshold_token(threshold)
                report[f"AP_{name}_{token}"] = metrics.ap
                report[f"num_match_{name}_{token}"] = float(curve.num_match)
                report[f"max_f1_{name}_{token}"] = metrics.max_f1
                report[f"optimal_conf_{name}_{token}"] = metrics.optimal_conf
                report[f"optimal_recall_{name}_{token}"] = metrics.optimal_recall
                report[f"optimal_precision_{name}_{token}"] = metrics.optimal_precision
        return report
