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

"""Ground-truth and prediction box counts per class."""

from __future__ import annotations

from collections import Counter

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.matching import Detection3DState
from autoware_ml.metrics.report import MetricReport


class BoxCounts(Metric[Detection3DState]):
    """Number of ground-truth and predicted boxes, in total and per class.

    The counts are taken from the state the suite hands over, so they are the
    boxes the other metrics actually score: ground truth is already filtered by
    ``min_num_points`` and ``max_distances`` at accumulation time, and both
    ground truth and predictions are clipped to the range the suite is currently
    evaluating. The suite appends the range suffix to every key, so each range
    reports its own counts.
    """

    def evaluate(self, state: Detection3DState, stage: EvalStage) -> MetricReport:
        """Count the accumulated ground-truth and predicted boxes.

        Args:
            state: Detection state holding the filtered, range-clipped samples.
            stage: Evaluation stage requesting the metrics.

        Returns:
            The total and per-class box counts.
        """
        gt_counts: Counter[int] = Counter()
        pred_counts: Counter[int] = Counter()
        for sample in state.samples:
            gt_counts.update(int(label) for label in sample.gt_labels.reshape(-1).tolist())
            pred_counts.update(int(label) for label in sample.pred_labels.reshape(-1).tolist())

        report = MetricReport()
        # Totals cover every label, including predicted labels outside the configured classes
        report.record("total_num_gts", float(sum(gt_counts.values())))
        report.record("total_num_preds", float(sum(pred_counts.values())))
        for label in state.labels(full=True):
            name = label_metric_name(label, state.class_names)
            report.record(f"num_gts_{name}", float(gt_counts.get(label, 0)))
            report.record(f"num_preds_{name}", float(pred_counts.get(label, 0)))
        return report
