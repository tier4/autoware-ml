"""Shared eval-output builder for 3D detection models.

Every detection model decodes its head into per-sample predictions and pairs
them with the ground-truth boxes and labels. This helper builds the flat
eval-output dict that :class:`~autoware_ml.metrics.detection3d.suite.AutowareDetection3DMetrics`
reads, so each model's ``build_eval_output`` is a one-line delegation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def detection_eval_output(
    predictions: list[dict[str, Any]], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair decoded predictions with ground truth for the detection metric.

    Args:
        predictions: Per-sample prediction dicts with ``bboxes_3d``,
            ``scores_3d``, and ``labels_3d``, as returned by ``bbox_head.predict``.
        batch: The batch dictionary holding the ground-truth boxes and labels.

    Returns:
        Flat eval-output dict consumed by the detection metric.
    """
    return {
        "predictions": predictions,
        "gt_boxes": batch["gt_boxes"],
        "gt_labels": batch["gt_labels"],
        "gt_num_points": batch.get("gt_num_points"),
        "gt_attributes": batch.get("gt_attributes"),
    }
