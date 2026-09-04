"""Heading-flip rate.

Corner displacement forgives a 180-degree heading flip: the box outline
barely moves under the cyclic corner assignment, so a reversed vehicle scores a
near-zero corner error. Yet a flipped heading inverts the object's velocity
direction and breaks tracking, so it must be penalized on its own. This metric
isolates it: every true positive whose absolute wrapped yaw error exceeds
``flip_threshold`` (a quarter turn) counts as one flip, a uniform +1 penalty
regardless of how far past the threshold. The suite's range loop bins it by
distance and it is reported per class.
"""

from __future__ import annotations

from math import pi

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import DetectionState, mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class HeadingFlipRate(Metric[DetectionState]):
    """Per-class fraction (and count) of true positives with a reversed heading."""

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        flip_threshold: float = pi / 2.0,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Store the match and flip thresholds.

        Args:
            tp_threshold: Center-distance match threshold in meters the errors are read at.
            flip_threshold: Absolute wrapped yaw error in radians above which a true positive
                counts as flipped.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.tp_threshold = float(tp_threshold)
        self.flip_threshold = float(flip_threshold)

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Report per-class heading-flip rate and count at the TP threshold.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        report: dict[str, float] = {}
        rates: list[float] = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            errors = curve.orientation_error[curve.true_positive]
            errors = errors[~np.isnan(errors)]
            name = label_metric_name(label, state.class_names)
            if errors.shape[0] == 0:
                report[f"flip_rate_{name}"] = float("nan")
                report[f"flip_count_{name}"] = 0.0
                continue
            flips = float(np.sum(errors > self.flip_threshold))
            rate = flips / float(errors.shape[0])
            report[f"flip_count_{name}"] = flips
            report[f"flip_rate_{name}"] = rate
            rates.append(rate)

        report["mflip_rate"] = mean_valid(rates)
        return report
