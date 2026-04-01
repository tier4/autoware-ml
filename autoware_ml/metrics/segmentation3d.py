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

"""Shared metrics for 3D semantic segmentation tasks.

Metric names use a consistent prefix convention:

* ``point_`` - **micro** metrics computed over all valid points at once.
* ``mean_``  - **macro** metrics computed per class and then averaged.
  Only classes with at least one ground-truth point contribute to the mean.
"""

from __future__ import annotations

import torch


def compute_point_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor | None:
    """Compute per-point accuracy while skipping ignored targets.

    Args:
        predictions: Predicted class indices ``(N,)``.
        targets: Ground-truth class indices ``(N,)``.
        ignore_index: Label value excluded from the computation.

    Returns:
        Scalar accuracy tensor, or ``None`` when every target is ignored.
    """
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return None
    return (predictions[valid_mask] == targets[valid_mask]).float().mean()


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> dict[str, torch.Tensor] | None:
    """Compute step-level segmentation metrics from predictions and targets.

    Returns both a micro (per-point) accuracy and a full set of macro
    (mean-across-classes) metrics derived from per-class TP / FP / FN / TN
    counts.  Classes without ground-truth support are excluded from the macro
    averages.

    Args:
        predictions: Predicted class indices ``(N,)``.
        targets: Ground-truth class indices ``(N,)``.
        num_classes: Total number of trainable classes.
        ignore_index: Label value excluded from the computation.

    Returns:
        Dictionary of named metric tensors, or ``None`` when every target is
        ignored.
    """
    valid = targets != ignore_index
    if not valid.any():
        return None

    pred = predictions[valid].long()
    tgt = targets[valid].long()
    total_valid = pred.numel()

    # -- micro metric (per-point) ------------------------------------------
    point_accuracy = (pred == tgt).float().sum() / total_valid

    # per-class counts via temporary confusion matrix
    encoded = tgt * num_classes + pred
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confmat = counts.reshape(num_classes, num_classes).float()

    tp = confmat.diag()
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    # Only average over classes that appear in the ground truth.
    support = confmat.sum(dim=1)
    has_support = support > 0

    if not has_support.any():
        return {"point_accuracy": point_accuracy}

    # macro metrics (mean across classes with support)
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
