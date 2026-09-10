"""Signed nearest-surface error.

Stopping distance is computed against the nearest face of an object, not its
center. The sign is the safety signal: a positive error means the predicted near
face sits farther than the truth (ego brakes late), negative means nearer
(over-caution). Reported per class at the nuScenes TP threshold, and the suite's
range loop bins it by distance.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import DetectionState, mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class NearestSurfaceError(Metric[DetectionState]):
    """Signed near-face error tails per class (meters).

    Reports the low/high signed percentiles (the over-caution and late-braking
    tails), the worst absolute error, and the signed mean bias.
    """

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the tail percentiles.

        Args:
            tp_threshold: Center-distance match threshold in meters the errors are read at.
            low_percentile: Signed percentile reported as the over-caution tail.
            high_percentile: Signed percentile reported as the late-braking tail.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.tp_threshold = float(tp_threshold)
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError("expected 0 <= low_percentile < high_percentile <= 100.")

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Report per-class signed nearest-surface error tails at the TP threshold.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        report: dict[str, float] = {}
        per_class_high: list[float] = []
        per_class_absmax: list[float] = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            errors = curve.nearest_surface_error[curve.true_positive]
            errors = errors[~np.isnan(errors)]
            name = label_metric_name(label, state.class_names)
            if errors.shape[0] == 0:
                report[f"nsurf_mean_{name}"] = float("nan")
                report[f"nsurf_low_{name}"] = float("nan")
                report[f"nsurf_high_{name}"] = float("nan")
                report[f"nsurf_absmax_{name}"] = float("nan")
                continue
            high = float(np.percentile(errors, self.high_percentile))
            absmax = float(np.max(np.abs(errors)))
            report[f"nsurf_mean_{name}"] = float(np.mean(errors))
            report[f"nsurf_low_{name}"] = float(np.percentile(errors, self.low_percentile))
            report[f"nsurf_high_{name}"] = high
            report[f"nsurf_absmax_{name}"] = absmax
            per_class_high.append(high)
            per_class_absmax.append(absmax)

        report["mnsurf_high"] = mean_valid(per_class_high)
        report["mnsurf_absmax"] = mean_valid(per_class_absmax)
        return report
