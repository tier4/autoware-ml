"""Ground-truth and prediction box counts per class."""

from __future__ import annotations

from collections import Counter

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.matching import DetectionState


class BoxCounts(Metric[DetectionState]):
    """Number of ground-truth and predicted boxes, in total and per class.

    The counts are taken from the state the suite hands over, so they are the
    boxes the other metrics actually score: ground truth is already filtered by
    ``min_num_points`` and ``eval_class_range`` at accumulation time, and both
    ground truth and predictions are clipped to the range the suite is currently
    evaluating. The suite appends the range suffix to every key, so each range
    reports its own counts.
    """

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Count the accumulated ground-truth and predicted boxes.

        Args:
            state: Detection state holding the filtered, range-clipped samples.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of total and per-class box counts.
        """
        gt_counts: Counter[int] = Counter()
        pred_counts: Counter[int] = Counter()
        for sample in state.samples:
            gt_counts.update(int(label) for label in sample.gt_labels.reshape(-1).tolist())
            pred_counts.update(int(label) for label in sample.pred_labels.reshape(-1).tolist())

        # Totals cover every label, including predicted labels outside the configured classes
        report = {
            "total_num_gts": float(sum(gt_counts.values())),
            "total_num_preds": float(sum(pred_counts.values())),
        }
        for label in state.labels(full=True):
            name = label_metric_name(label, state.class_names)
            report[f"num_gts_{name}"] = float(gt_counts.get(label, 0))
            report[f"num_preds_{name}"] = float(pred_counts.get(label, 0))
        return report
