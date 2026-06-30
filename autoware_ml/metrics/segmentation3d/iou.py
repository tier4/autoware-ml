"""Intersection-over-union metric."""

from __future__ import annotations

import torch

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.confusion import (
    ConfusionState,
    class_name_token,
    macro,
)


class IoU(Metric[ConfusionState]):
    """Macro mean IoU in every stage. Test adds the frequency-weighted IoU and the
    per-class IoU for each class with ground-truth support.
    """

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> dict[str, float]:
        union = state.predicted + state.actual - state.true_positive
        zeros = torch.zeros_like(state.true_positive)
        iou = torch.where(union > 0, state.true_positive / union, zeros)

        report = {"mIoU": macro(iou, state.has_support)}
        if stage is not EvalStage.TEST:
            return report

        if bool(state.has_support.any()):
            report["fwIoU"] = float((state.frequency * iou)[state.has_support].sum().item())
        else:
            report["fwIoU"] = float("nan")
        for class_index in range(state.num_classes):
            if not bool(state.has_support[class_index]):
                continue
            report[f"iou_{class_name_token(class_index, state.class_names)}"] = float(
                iou[class_index].item()
            )
        return report
