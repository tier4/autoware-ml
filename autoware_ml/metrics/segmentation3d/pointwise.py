"""Step-level pointwise segmentation helpers.

These compute metrics directly from one batch of point predictions and targets,
independent of the accumulating suite. They are handy for quick per-step checks.
Metric names follow the convention: ``point_`` for micro metrics over all valid
points, ``mean_`` for macro metrics averaged over classes with support.
"""

from __future__ import annotations

import torch


def compute_point_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Per-point accuracy over valid points, skipping ignored targets."""
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        raise ValueError("Cannot compute point accuracy: every target is ignored.")
    return (predictions[valid_mask] == targets[valid_mask]).float().mean()


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> dict[str, torch.Tensor]:
    """Micro accuracy plus macro IoU/precision/recall/F1 for one batch.

    Classes without ground-truth support are excluded from the macro averages.
    """
    valid = targets != ignore_index
    if not valid.any():
        raise ValueError("Cannot compute segmentation metrics: every target is ignored.")

    pred = predictions[valid].long()
    tgt = targets[valid].long()
    total_valid = pred.numel()

    point_accuracy = (pred == tgt).float().sum() / total_valid

    encoded = tgt * num_classes + pred
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confmat = counts.reshape(num_classes, num_classes).float()

    tp = confmat.diag()
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    support = confmat.sum(dim=1)
    has_support = support > 0

    if not has_support.any():
        return {"point_accuracy": point_accuracy}

    iou = tp / (tp + fp + fn).clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)

    return {
        "point_accuracy": point_accuracy,
        "mean_iou": iou[has_support].mean(),
        "mean_precision": precision[has_support].mean(),
        "mean_recall": recall[has_support].mean(),
        "mean_f1": f1[has_support].mean(),
    }
