"""Eval output builder shared by the 3D detection models.

The detection metric reads plain dicts, so this helper turns the typed predictions of a batch
into dicts and pairs them with the ground truth boxes and labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions


def detection_eval_output(
    predictions: Sequence[Detection3DSamplePredictions], batch_inputs_dict: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair decoded predictions with ground truth for the detection metric.

    Args:
        predictions: Per sample typed predictions as returned by the head's predict.
        batch_inputs_dict: Model inputs holding the ground truth boxes and labels.

    Returns:
        Flat eval output dict consumed by the detection metric.
    """
    if "gt_boxes" not in batch_inputs_dict or "gt_labels" not in batch_inputs_dict:
        raise ValueError("The detection eval output requires ground truth boxes and labels.")
    eval_out: dict[str, Any] = {
        "predictions": [
            {
                "bboxes_3d": prediction.bboxes_3d,
                "scores_3d": prediction.scores_3d,
                "labels_3d": prediction.labels_3d,
            }
            for prediction in predictions
        ],
        "gt_boxes": list(batch_inputs_dict["gt_boxes"]),
        "gt_labels": list(batch_inputs_dict["gt_labels"]),
    }
    # Copy the per frame point counts and metadata (ego pose, scene token) when the dataset
    # provides them. A filter that needs a missing key raises in its suite.
    for key in ("gt_num_points", "ego2global", "scene_token"):
        if key in batch_inputs_dict:
            eval_out[key] = list(batch_inputs_dict[key])
    return eval_out


def typed_detection_eval_output(
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

    if batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
        raise ValueError("The detection eval output requires a 3D detection ground truth batch.")

    gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
    # Read the valid counts back to the host once. They index every ground truth tensor below,
    # and gt_valid_bboxes can live on the GPU.
    valid = gt_detections.gt_valid_bboxes.tolist()
    if len(predictions.detection3d_predictions) != len(valid):
        raise ValueError(
            "The predictions must hold one entry per sample, got "
            f"{len(predictions.detection3d_predictions)} predictions for {len(valid)} samples."
        )

    batch: dict[str, Any] = {
        "gt_boxes": [gt_detections.gt_bboxes_3d[i, : valid[i]] for i in range(len(valid))],
        "gt_labels": [gt_detections.gt_labels_3d[i, : valid[i]] for i in range(len(valid))],
        "gt_num_points": [
            gt_detections.gt_bboxes_num_points[i, : valid[i]] for i in range(len(valid))
        ],
    }
    # Per frame metadata for the region and collision filters, when the dataset attached it.
    frame_meta_batch = batch_inputs.multi_task_gt_batch.frame_meta_batch
    if frame_meta_batch is not None:
        batch["ego2global"] = list(frame_meta_batch.ego2globals)
        batch["scene_token"] = list(frame_meta_batch.scene_tokens)

    return detection_eval_output(
        predictions=predictions.detection3d_predictions, batch_inputs_dict=batch
    )
