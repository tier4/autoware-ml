"""Corner displacement error.

Corner displacement couples position, size, and yaw into one distance in meters,
measured where planning collides with things: the box outline. It also exposes
footprint inflation that center distance is blind to. Reported per class at the
nuScenes TP threshold, and the suite's range loop bins it by distance.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import DetectionState, mean_valid
from autoware_ml.metrics.base import number_token
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class CornerError(Metric[DetectionState]):
    """Mean / p95 / max true-positive corner displacement, per class (meters)."""

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        percentiles: tuple[float, ...] = (95.0,),
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the reported percentiles.

        Args:
            tp_threshold: Center-distance match threshold in meters the errors are read at.
            percentiles: Percentiles in [0, 100] reported next to the mean and max.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.tp_threshold = float(tp_threshold)
        self.percentiles = tuple(float(p) for p in percentiles)
        for p in self.percentiles:
            if not 0.0 <= p <= 100.0:
                raise ValueError(f"percentile {p} outside [0, 100].")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Report per-class corner-displacement statistics at the TP threshold.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        report: dict[str, float] = {}
        per_class_mean: list[float] = []
        per_class_max: list[float] = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            errors = curve.corner_error[curve.true_positive]
            errors = errors[~np.isnan(errors)]
            name = label_metric_name(label, state.class_names)
            if errors.shape[0] == 0:
                report[f"corner_mean_{name}"] = float("nan")
                report[f"corner_max_{name}"] = float("nan")
                for p in self.percentiles:
                    report[f"corner_p{number_token(p)}_{name}"] = float("nan")
                continue
            mean_value = float(np.mean(errors))
            maximum = float(np.max(errors))
            report[f"corner_mean_{name}"] = mean_value
            report[f"corner_max_{name}"] = maximum
            per_class_mean.append(mean_value)
            per_class_max.append(maximum)
            for p in self.percentiles:
                report[f"corner_p{number_token(p)}_{name}"] = float(np.percentile(errors, p))

        report["mcorner_mean"] = mean_valid(per_class_mean)
        report["mcorner_max"] = mean_valid(per_class_max)
        return report
