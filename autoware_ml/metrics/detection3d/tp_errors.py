"""True-positive error metric (test only by default)."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import (
    curve_metrics,
    mean_tp_errors,
    select_optimal_tp_errors,
    select_recall_tp_errors,
)
from autoware_ml.metrics.detection3d.naming import label_metric_name, threshold_token
from autoware_ml.metrics.detection3d.structures import (
    ERROR_NAMES,
    DetectionState,
    SelectedTpErrors,
)


class TpErrors(Metric[DetectionState]):
    """Mean ATE/AOE/ASE/AVE/AAE per recall variant, plus the per-class
    per-threshold errors and their match counts. Variants are the configured
    recall targets plus the optimal-F1 operating point.
    """

    def __init__(
        self,
        recall_targets: dict[str, float] | None = None,
        stages: tuple[str, ...] | list[str] = ("test",),
    ) -> None:
        super().__init__(stages)
        self.recall_targets = (
            {"default": 0.10, "medium": 0.40}
            if recall_targets is None
            else {str(name): float(value) for name, value in recall_targets.items()}
        )

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Compute true-positive error summaries for detection predictions.

        Args:
            state: Detection state with cached match curves.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of aggregate and per-class TP error metrics.
        """
        labels = state.labels(full=True)
        variants: dict[str, dict[tuple[int, float], SelectedTpErrors]] = {
            name: {} for name in self.recall_targets
        }
        variants["optimal"] = {}

        for label in labels:
            for threshold in state.thresholds:
                curve = state.match_curve(label, threshold)
                for name, target in self.recall_targets.items():
                    variants[name][(label, threshold)] = select_recall_tp_errors(curve, target)
                optimal_index = curve_metrics(curve).optimal_index
                variants["optimal"][(label, threshold)] = select_optimal_tp_errors(
                    curve, optimal_index
                )

        report: dict[str, float] = {}
        for variant_name, selected in variants.items():
            # Average only over curves that actually selected true positives;
            # classes with no GT (or too few for the recall bucket) have no error
            # to measure and must not pull the mean to the worst case.
            kept = [item.errors for item in selected.values() if item.count > 0]
            mean_errors = mean_tp_errors(kept) if kept else {name: 1.0 for name in ERROR_NAMES}
            for error_name, value in mean_errors.items():
                report[f"m{error_name}_{variant_name}"] = value

        for variant_name, selected in variants.items():
            for (label, threshold), item in selected.items():
                name = label_metric_name(label, state.class_names)
                token = threshold_token(threshold)
                report[f"tp_error_num_match_{name}_{variant_name}_{token}"] = float(item.count)
                for error_name, value in item.errors.items():
                    report[f"{error_name}_{name}_{variant_name}_{token}"] = value
        return report
