"""Segmentation calibration error.

A model that says "90% road" should be right about 90% of the time, and confident
wrong predictions are the dangerous ones. Expected Calibration Error bins points by
predicted confidence (max softmax) and sums the gap between confidence and
accuracy. Reported overall and macro-averaged per class, so the dominant road
class does not hide poor calibration on rare classes.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.point_cloud import (
    PointCloudSegState,
    valid_point_mask,
)


def _ece_from_bins(conf_sum: np.ndarray, correct_sum: np.ndarray, count: np.ndarray) -> float:
    total = count.sum()
    if total == 0:
        return float("nan")
    nonempty = count > 0
    accuracy = correct_sum[nonempty] / count[nonempty]
    mean_conf = conf_sum[nonempty] / count[nonempty]
    weight = count[nonempty] / total
    return float(np.sum(weight * np.abs(accuracy - mean_conf)))


class CalibrationError(Metric[PointCloudSegState]):
    """Expected Calibration Error, overall and macro-averaged per class.

    Standard classwise ECE: the per-class rows bin by the point's **predicted**
    class (the class whose confidence was reported), the overall ECE over all
    points. A class the model never predicts contributes no row.
    """

    def __init__(
        self,
        num_bins: int = 15,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the bin count.

        Args:
            num_bins: Number of equal-width confidence bins.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.num_bins = int(num_bins)
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1.")

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Bin per-class confidence vs accuracy across frames and reduce to ECE.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        num_classes = state.num_classes
        conf_sum = np.zeros((num_classes, self.num_bins), dtype=np.float64)
        correct_sum = np.zeros((num_classes, self.num_bins), dtype=np.float64)
        count = np.zeros((num_classes, self.num_bins), dtype=np.float64)
        for frame in state.frames:
            valid = valid_point_mask(frame, num_classes, state.ignore_index)
            target = frame.target[valid]
            pred = frame.pred[valid]
            if target.shape[0] == 0:
                continue
            conf = frame.confidence[valid]
            correct = (pred == target).astype(np.float64)
            bin_index = np.clip((conf * self.num_bins).astype(np.int64), 0, self.num_bins - 1)
            np.add.at(conf_sum, (pred, bin_index), conf)
            np.add.at(correct_sum, (pred, bin_index), correct)
            np.add.at(count, (pred, bin_index), 1.0)

        overall = _ece_from_bins(conf_sum.sum(0), correct_sum.sum(0), count.sum(0))
        per_class = [
            _ece_from_bins(conf_sum[c], correct_sum[c], count[c])
            for c in range(num_classes)
            if count[c].sum() > 0
        ]
        macro = float(np.mean(per_class)) if per_class else float("nan")
        return {"ece": overall, "ece_macro": macro}
