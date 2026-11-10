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

"""Overall accuracy metric."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.report import MetricReport
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class Accuracy(Metric[ConfusionState]):
    """Micro point accuracy: correct points over all valid points. A single global
    ratio, so there is no per-class breakdown.
    """

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> MetricReport:
        """Compute global point accuracy from a confusion state.

        Args:
            state: Segmentation confusion state.
            stage: Evaluation stage requesting the metric.

        Returns:
            The accuracy.
        """
        report = MetricReport()
        if state.total > 0:
            report.record("acc", float((state.true_positive.sum() / state.total).item()))
        else:
            report.record("acc", float("nan"))
        return report
