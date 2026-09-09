"""Critical FP / FN over the ego reachable set.

The two counts are reported separately because they mean different things: a
false positive in ego's path causes phantom braking, a false negative in ego's
path means driving toward something nobody saw. "In ego's path" is the one
reachability model: an object is critical
when it can collide with ego within the horizon, i.e. its per-box TTC (computed
by the suite's collision provider, which caps finite values at its horizon) is
finite. Both are reported as a function of the confidence threshold, because
release engineering picks a threshold and needs to know whether any threshold is
acceptable on both axes at once.

Matching is class-agnostic (a box detected where an object is avoids both a
phantom and a miss), and only the unmatched boxes that are critical are counted.
Frames whose scene has no lanelet map carry no TTC and are excluded from the
per-frame denominator (the suite logs that coverage).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric, number_token
from autoware_ml.metrics.detection3d.criticality import greedy_match
from autoware_ml.metrics.detection3d.matching import DetectionState
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class CriticalFPFN(Metric[DetectionState]):
    """Per-frame false positives / negatives in the ego reachable set, per confidence."""

    needs_ttc = True

    def __init__(
        self,
        confidences: tuple[float, ...] = (0.3, 0.5),
        match_threshold: float = DEFAULT_TP_THRESHOLD,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the confidence operating points and the match threshold.

        Args:
            confidences: Score floors in ``[0, 1]``, the FP and FN counts are reported at each.
            match_threshold: Class-agnostic center-distance match threshold in meters.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.confidences = tuple(float(c) for c in confidences)
        self.match_threshold = float(match_threshold)
        if not self.confidences or any(not 0.0 <= c <= 1.0 for c in self.confidences):
            raise ValueError("confidences must name at least one score floor in [0, 1].")
        if self.match_threshold <= 0.0:
            raise ValueError("match_threshold must be > 0.")
        # Predictions below the lowest operating point are never counted here, so
        # the suite does not have to propagate them.
        self.ttc_score_floor = min(self.confidences)

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Count critical FP / FN (finite TTC) at each confidence threshold.

        Args:
            state: Detection state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        covered_frames = 0
        fp_total = {conf: 0 for conf in self.confidences}
        fn_total = {conf: 0 for conf in self.confidences}
        fp_class: Counter[tuple[float, int]] = Counter()
        fn_class: Counter[tuple[float, int]] = Counter()
        for sample in state.samples:
            if sample.gt_ttc is None or sample.pred_ttc is None:
                raise ValueError(
                    "CriticalFPFN needs per-box TTC, build the suite with a collision provider."
                )
            if not sample.ttc_covered:
                continue
            covered_frames += 1
            gt_boxes = sample.gt_boxes.numpy().astype(np.float64)
            gt_labels = sample.gt_labels.numpy().astype(np.int64)
            gt_critical = np.isfinite(sample.gt_ttc.numpy())
            pred_boxes = sample.pred_boxes.numpy().astype(np.float64)
            pred_scores = sample.pred_scores.numpy().astype(np.float64)
            pred_labels = sample.pred_labels.numpy().astype(np.int64)
            pred_ttc = sample.pred_ttc.numpy()
            for conf in self.confidences:
                keep = pred_scores >= conf
                kept_boxes = pred_boxes[keep]
                kept_scores = pred_scores[keep]
                kept_labels = pred_labels[keep]
                kept_ttc = pred_ttc[keep]
                if np.isnan(kept_ttc).any():
                    raise ValueError(
                        "prediction TTC is missing at this confidence: the suite skipped it "
                        "below its score floor, which must be the lowest confidence read here."
                    )
                kept_critical = np.isfinite(kept_ttc)
                is_tp, matched_gt = greedy_match(
                    gt_boxes[:, :2], kept_boxes[:, :2], kept_scores, self.match_threshold
                )
                # Critical false positives: unmatched predictions that are in-path.
                false_positive = (~is_tp) & kept_critical
                matched = np.zeros(gt_boxes.shape[0], dtype=bool)
                matched[matched_gt[is_tp]] = True
                # Critical false negatives: unmatched ground truth that is in-path.
                false_negative = (~matched) & gt_critical
                fp_total[conf] += int(false_positive.sum())
                fn_total[conf] += int(false_negative.sum())
                fp_class.update((conf, int(label)) for label in kept_labels[false_positive])
                fn_class.update((conf, int(label)) for label in gt_labels[false_negative])

        # No covered frame means the slice has no basis: report NaN, never a
        # fake zero (a dataset without maps would otherwise look perfectly safe).
        denominator = covered_frames if covered_frames else float("nan")
        report: dict[str, float] = {}
        # A confidence is not a distance, so it gets its own token, never the metre suffix.
        for conf in self.confidences:
            token = f"conf{number_token(conf)}"
            report[f"critical_fp_{token}"] = fp_total[conf] / denominator
            report[f"critical_fn_{token}"] = fn_total[conf] / denominator
        if stage is EvalStage.TEST:
            for label in state.labels(full=True):
                name = label_metric_name(label, state.class_names)
                for conf in self.confidences:
                    token = f"conf{number_token(conf)}"
                    report[f"critical_fp_{name}_{token}"] = fp_class[(conf, label)] / denominator
                    report[f"critical_fn_{name}_{token}"] = fn_class[(conf, label)] / denominator
        return report
