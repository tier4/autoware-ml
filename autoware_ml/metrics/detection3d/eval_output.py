"""Eval output builder shared by the 3D detection models.

The detection metric reads plain dicts, so this helper turns the typed predictions of a batch
into dicts and pairs them with the typed ground truth boxes and labels.
"""

from __future__ import annotations

from typing import Any

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions


def detection_eval_output(
    predictions: ModelPredictions, batch_inputs: ModelBatchInputs
) -> dict[str, Any]:
    """Pair the typed predictions of a batch with its typed ground truth for the detection metric.

    Args:
        predictions: Decoded predictions of the batch.
        batch_inputs: Batch inputs holding the ground truth boxes and labels.

    Returns:
        Flat eval output dict consumed by the detection metric.
    """
    if predictions.detection3d_predictions is None:
        raise ValueError("The detection eval output requires 3D detection predictions.")

    gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
    if gt_detections is None:
        raise ValueError("The detection eval output requires a 3D detection ground truth batch.")

    gt_boxes = gt_detections.valid_bboxes_3d()
    if len(predictions.detection3d_predictions) != len(gt_boxes):
        raise ValueError(
            "The predictions must hold one entry per sample, got "
            f"{len(predictions.detection3d_predictions)} predictions for {len(gt_boxes)} samples."
        )

    eval_out: dict[str, Any] = {
        "predictions": [
            {
                "bboxes_3d": prediction.bboxes_3d,
                "scores_3d": prediction.scores_3d,
                "labels_3d": prediction.labels_3d,
            }
            for prediction in predictions.detection3d_predictions
        ],
        "gt_boxes": gt_boxes,
        "gt_labels": gt_detections.valid_labels_3d(),
        "gt_num_points": gt_detections.valid_bboxes_num_points(),
    }
    # Per frame metadata for the region and collision filters, when the dataset attached it.
    frame_meta_batch = batch_inputs.multi_task_gt_batch.frame_meta_batch
    if frame_meta_batch is not None:
        eval_out["ego2global"] = list(frame_meta_batch.ego2globals)
        eval_out["scene_token"] = list(frame_meta_batch.scene_tokens)
    return eval_out
