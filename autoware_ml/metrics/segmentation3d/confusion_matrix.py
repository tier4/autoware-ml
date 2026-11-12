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

"""Segmentation confusion matrix (point level).

The confusion suite already accumulates the point confusion matrix (rows = true
class, columns = predicted class) as its state, folded onto behaviour groups in a
grouped suite, so this metric is a thin reader that emits its cells as
``confusion_<true>__<pred>`` counts.
"""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.confusion_report import confusion_cells
from autoware_ml.metrics.report import MetricReport
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class ConfusionMatrix(Metric[ConfusionState]):
    """Point true vs. predicted class counts, read straight from the suite's matrix."""

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> MetricReport:
        """Emit the accumulated confusion matrix as flat count keys.

        Args:
            state: Confusion state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            This metric's report.
        """
        return confusion_cells(state.confusion.cpu().numpy(), state.class_names)
