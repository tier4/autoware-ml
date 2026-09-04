"""True-positive error metric (test only by default)."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import (
    DetectionState,
    mean_tp_errors,
    select_optimal_tp_errors,
    select_recall_tp_errors,
)
from autoware_ml.metrics.detection3d.naming import label_metric_name, threshold_token
from autoware_ml.metrics.detection3d.structures import (
    DEFAULT_TP_THRESHOLD,
    DEFAULT_MATCH_THRESHOLDS,
    ERROR_NAMES,
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
        thresholds: tuple[float, ...] = DEFAULT_MATCH_THRESHOLDS,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the recall variants and the operating-point threshold.

        Args:
            recall_targets: Variant name to recall level in [0, 1]. Defaults to
                ``default: 0.10`` and ``medium: 0.40``. The name ``optimal`` is reserved for the
                max-F1 operating point that is always reported.
            thresholds: Center-distance match thresholds in meters for the detail keys.
            tp_threshold: The single threshold the aggregate means are computed at, must be one
                of ``thresholds``.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        if not self.thresholds:
            raise ValueError("thresholds must not be empty.")
        # Aggregate means use the single nuScenes operating point. The per-class
        # per-threshold detail keys still cover the full threshold set.
        self.tp_threshold = float(tp_threshold)
        if self.tp_threshold not in self.thresholds:
            raise ValueError(
                f"tp_threshold={self.tp_threshold} is not one of thresholds={self.thresholds}, "
                "the aggregate operating point must be an explicitly configured threshold."
            )
        self.recall_targets = (
            {"default": 0.10, "medium": 0.40}
            if recall_targets is None
            else {str(name): float(value) for name, value in recall_targets.items()}
        )
        if "optimal" in self.recall_targets:
            raise ValueError('"optimal" is the reserved max-F1 operating point, pick another name.')
        for name, value in self.recall_targets.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"recall target {name!r} must be in [0, 1], got {value}.")

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
            for threshold in self.thresholds:
                curve = state.match_curve(label, threshold)
                for name, target in self.recall_targets.items():
                    variants[name][(label, threshold)] = select_recall_tp_errors(curve, target)
                optimal_index = state.curve_metrics(label, threshold).optimal_index
                variants["optimal"][(label, threshold)] = select_optimal_tp_errors(
                    curve, optimal_index
                )

        report: dict[str, float] = {}
        # Aggregate means: one entry per class, selected at the single
        # tp_threshold operating point (averaging over thresholds would weight
        # easy classes 4x). Selected directly from that threshold's curve, so
        # the aggregates stay well-defined whatever detail set is configured;
        # only classes that actually selected true positives contribute (a class
        # absent here, or with too few GT for the recall bucket, has no error to
        # measure).
        for variant_name in variants:
            kept = []
            for label in labels:
                curve = state.match_curve(label, self.tp_threshold)
                if variant_name == "optimal":
                    optimal_index = state.curve_metrics(label, self.tp_threshold).optimal_index
                    item = select_optimal_tp_errors(curve, optimal_index)
                else:
                    item = select_recall_tp_errors(curve, self.recall_targets[variant_name])
                if item.count > 0:
                    kept.append(item.errors)
            mean_errors = (
                mean_tp_errors(kept) if kept else {name: float("nan") for name in ERROR_NAMES}
            )
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
