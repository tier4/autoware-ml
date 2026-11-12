# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""True-positive error metric (test only by default)."""

from __future__ import annotations

from collections.abc import Sequence

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import (
    Detection3DState,
    mean_tp_errors,
    select_optimal_tp_errors,
    select_recall_tp_errors,
)
from autoware_ml.metrics.detection3d.naming import label_metric_name, threshold_token
from autoware_ml.metrics.detection3d.structures import (
    DEFAULT_MATCH_THRESHOLDS,
    DEFAULT_TP_THRESHOLD,
    RecallTarget,
    SelectedTpErrors,
    TpErrorSet,
)
from autoware_ml.metrics.report import MetricReport

DEFAULT_RECALL_TARGETS = (
    RecallTarget(name="default", recall=0.10),
    RecallTarget(name="medium", recall=0.40),
)
# The max-F1 operating point, reported next to the recall targets as the ``optimal`` variant.
OPTIMAL_VARIANT = "optimal"


class TpErrors(Metric[Detection3DState]):
    """Mean ATE/AOE/ASE/AVE/AAE per recall variant, plus the per-class
    per-threshold errors and their match counts. Variants are the configured
    recall targets plus the optimal-F1 operating point.
    """

    def __init__(
        self,
        recall_targets: Sequence[RecallTarget] | None = None,
        thresholds: tuple[float, ...] = DEFAULT_MATCH_THRESHOLDS,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the recall variants and the operating-point threshold.

        Args:
            recall_targets: Recall operating points, each named distinctly. Defaults to
                ``default`` at 0.10 and ``medium`` at 0.40. The ``optimal`` max-F1 operating
                point is always reported in addition.
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
            DEFAULT_RECALL_TARGETS if recall_targets is None else tuple(recall_targets)
        )
        names = [target.name for target in self.recall_targets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"recall target names must be distinct, got {duplicates} twice.")

    def _variants(self) -> list[tuple[str, RecallTarget | None]]:
        """The reported variants: the recall targets, then the optimal-F1 point as ``None``."""
        variants: list[tuple[str, RecallTarget | None]] = [
            (target.name, target) for target in self.recall_targets
        ]
        variants.append((OPTIMAL_VARIANT, None))
        return variants

    @staticmethod
    def _selected(
        state: Detection3DState, label: int, threshold: float, target: RecallTarget | None
    ) -> SelectedTpErrors:
        """The errors of one class at one threshold, selected up to a target or max F1."""
        curve = state.match_curve(label, threshold)
        if target is None:
            return select_optimal_tp_errors(
                curve, state.curve_metrics(label, threshold).optimal_index
            )
        return select_recall_tp_errors(curve, target.recall)

    def evaluate(self, state: Detection3DState, stage: EvalStage) -> MetricReport:
        """Compute true-positive error summaries for detection predictions.

        Args:
            state: Detection state with cached match curves.
            stage: Evaluation stage requesting the metrics.

        Returns:
            The aggregate and per-class TP errors.
        """
        labels = state.labels(full=True)
        variants = self._variants()
        report = MetricReport()

        # Aggregate means: one entry per class, selected at the single
        # tp_threshold operating point (averaging over thresholds would weight
        # easy classes 4x). Selected directly from that threshold's curve, so
        # the aggregates stay well-defined whatever detail set is configured;
        # only classes that actually selected true positives contribute (a class
        # absent here, or with too few GT for the recall bucket, has no error to
        # measure).
        for variant_name, target in variants:
            kept: list[TpErrorSet] = []
            for label in labels:
                item = self._selected(state, label, self.tp_threshold, target)
                if item.count > 0:
                    kept.append(item.errors)
            mean_errors = mean_tp_errors(kept) if kept else TpErrorSet.nan()
            for error_name, value in mean_errors.named():
                report.record(f"m{error_name}_{variant_name}", value)

        for variant_name, target in variants:
            for label in labels:
                name = label_metric_name(label, state.class_names)
                for threshold in self.thresholds:
                    item = self._selected(state, label, threshold, target)
                    token = threshold_token(threshold)
                    report.record(
                        f"tp_error_num_match_{name}_{variant_name}_{token}", float(item.count)
                    )
                    for error_name, value in item.errors.named():
                        report.record(f"{error_name}_{name}_{variant_name}_{token}", value)
        return report
