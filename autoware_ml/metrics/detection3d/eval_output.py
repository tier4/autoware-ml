"""Eval output builder shared by the 3D detection models.

Pairs the decoded predictions of a batch with the ground truth boxes and labels of every
sample, one :class:`~autoware_ml.metrics.eval_output.Detection3DEvalFrame` per sample.
"""

from __future__ import annotations

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.eval_output import Detection3DEvalFrame, Detection3DEvalOutput


def detection3d_eval_output(
    predictions: ModelPredictions, batch_inputs: ModelBatchInputs
) -> Detection3DEvalOutput:
    """Pair the predictions of a batch with its ground truth for the detection suite.

    Args:
        predictions: Decoded predictions of the batch.
        batch_inputs: Batch inputs holding the ground truth boxes and labels.

    Returns:
        One frame per sample of the batch, padding samples flagged.
    """
    if predictions.detection3d_predictions is None:
        raise ValueError("The detection eval output requires 3D detection predictions.")

    gt_batch = batch_inputs.multi_task_gt_batch
    gt_detections = gt_batch.detection3d_gt_batch
    if gt_detections is None:
        raise ValueError("The detection eval output requires a 3D detection ground truth batch.")

    gt_boxes = gt_detections.valid_bboxes_3d()
    gt_labels = gt_detections.valid_labels_3d()
    gt_num_points = gt_detections.valid_bboxes_num_points()
    num_frames = len(gt_boxes)
    if len(predictions.detection3d_predictions) != num_frames:
        raise ValueError(
            "The predictions must hold one entry per sample, got "
            f"{len(predictions.detection3d_predictions)} predictions for {num_frames} samples."
        )
    # Per frame metadata for the region and collision filters, when the dataset attached it.
    frame_meta_batch = gt_batch.frame_meta_batch
    if frame_meta_batch is not None and frame_meta_batch.ego2globals.shape[0] != num_frames:
        raise ValueError(
            f"The frame metadata carries {frame_meta_batch.ego2globals.shape[0]} ego2global "
            f"entries for {num_frames} samples, per-frame metadata must align one-to-one."
        )
    # A sample padding a streaming lane keeps its frame, since the metrics gather the frames
    # across the GPUs entry by entry; the suite leaves it out.
    padding = gt_batch.padding_mask.tolist()
    return Detection3DEvalOutput(
        frames=tuple(
            Detection3DEvalFrame(
                predictions=prediction,
                gt_boxes=gt_boxes[index],
                gt_labels=gt_labels[index],
                gt_num_points=gt_num_points[index],
                is_padding=bool(padding[index]),
                frame_meta=None if frame_meta_batch is None else frame_meta_batch.sample(index),
            )
            for index, prediction in enumerate(predictions.detection3d_predictions)
        )
    )
