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

"""Intersection-over-union metric."""

from __future__ import annotations

import torch

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.report import MetricReport
from autoware_ml.metrics.segmentation3d.confusion import (
    ConfusionState,
    class_name_token,
    macro,
)


class IoU(Metric[ConfusionState]):
    """Macro mean IoU in every stage. Test adds the frequency-weighted IoU and the
    per-class IoU for each class with ground-truth support.
    """

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> MetricReport:
        """Compute IoU metrics from a confusion state.

        Args:
            state: Segmentation confusion state.
            stage: Evaluation stage requesting the metrics.

        Returns:
            The IoU values.
        """
        union = state.predicted + state.actual - state.true_positive
        zeros = torch.zeros_like(state.true_positive)
        iou = torch.where(union > 0, state.true_positive / union, zeros)

        report = MetricReport()
        report.record("mIoU", macro(iou, state.has_support))
        if stage is not EvalStage.TEST:
            return report

        if bool(state.has_support.any()):
            report.record("fwIoU", float((state.frequency * iou)[state.has_support].sum().item()))
        else:
            report.record("fwIoU", float("nan"))
        for class_index in range(state.num_classes):
            if not bool(state.has_support[class_index]):
                continue
            report.record(
                f"iou_{class_name_token(class_index, state.class_names)}",
                float(iou[class_index].item()),
            )
        return report
