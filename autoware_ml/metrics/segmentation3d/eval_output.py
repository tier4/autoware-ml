"""Shared eval-output builder for the 3D segmentation suites.

Both segmentation suites consume one contract, a
:class:`~autoware_ml.metrics.eval_output.Segmentation3DEvalOutput` with one frame per
sample carrying that frame's points (coordinates, predicted and target labels, per-class
scores), the per-frame metadata the configured filters need and, for the cross-task
partial detection metric, the frame's detection ground truth boxes.

Metadata and boxes are copied from the batch when the dataset supplies them. A filter or
metric that needs a missing part fails loud through the suite's requirements, so absence
is never silent.
"""

from __future__ import annotations

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.metrics.eval_output import Segmentation3DEvalFrame, Segmentation3DEvalOutput


def segmentation3d_frames_eval_output(
    coord: torch.Tensor,
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    scores: torch.Tensor,
    frame_ids: torch.Tensor,
    num_frames: int,
    batch_inputs: ModelBatchInputs,
) -> Segmentation3DEvalOutput:
    """Split batch-concatenated per-point tensors into one frame per sample.

    Args:
        coord: ``(N, 3+)`` point coordinates (base_link).
        pred_labels: ``(N,)`` predicted class per point.
        target_labels: ``(N,)`` ground-truth class per point.
        scores: ``(N, C)`` softmax probabilities per point.
        frame_ids: ``(N,)`` frame index of every point.
        num_frames: Number of frames in the batch.
        batch_inputs: Model inputs of the batch. The frame metadata (ego pose, scene token,
            lanelet map path) and the detection ground truth are attached to each frame when
            the batch carries them.

    Returns:
        The frames of the batch.
    """
    gt_batch = batch_inputs.multi_task_gt_batch
    frame_meta_batch = gt_batch.frame_meta_batch
    if frame_meta_batch is not None and frame_meta_batch.ego2globals.shape[0] != num_frames:
        raise ValueError(
            f"The frame metadata carries {frame_meta_batch.ego2globals.shape[0]} ego2global "
            f"entries for {num_frames} frames, per-frame metadata must align one-to-one or "
            "points get another frame's context."
        )
    gt_boxes = None
    gt_box_labels = None
    if gt_batch.detection3d_gt_batch is not None:
        gt_boxes = gt_batch.detection3d_gt_batch.valid_bboxes_3d()
        gt_box_labels = gt_batch.detection3d_gt_batch.valid_labels_3d()
        if len(gt_boxes) != num_frames:
            raise ValueError(
                f"The detection ground truth holds {len(gt_boxes)} samples for {num_frames} "
                "frames, per-frame boxes must align one-to-one."
            )
    for name, tensor in (
        ("coord", coord),
        ("pred_labels", pred_labels),
        ("target_labels", target_labels),
        ("scores", scores),
    ):
        if tensor.shape[0] != frame_ids.numel():
            raise ValueError(
                f"{name} carries {tensor.shape[0]} rows for {frame_ids.numel()} frame ids, "
                "every per-point array must describe the same points."
            )
    # frame_ids come from cumulative point offsets, so they are non-decreasing.
    # One split then replaces a full O(frames x N) equality scan per frame.
    if frame_ids.numel() and not bool((frame_ids[1:] >= frame_ids[:-1]).all()):
        raise ValueError("frame_ids must be non-decreasing (per-frame point blocks).")
    if frame_ids.numel() and int(frame_ids.max()) >= num_frames:
        raise ValueError(
            f"frame_ids reach {int(frame_ids.max())} for {num_frames} frames, a malformed "
            "offset would silently drop the trailing frames' points."
        )
    counts = torch.bincount(frame_ids, minlength=num_frames).tolist()
    coord_split = torch.split(coord, counts)
    pred_split = torch.split(pred_labels, counts)
    target_split = torch.split(target_labels, counts)
    scores_split = torch.split(scores, counts)

    return Segmentation3DEvalOutput(
        frames=tuple(
            Segmentation3DEvalFrame(
                coord=coord_split[index],
                pred=pred_split[index],
                target=target_split[index],
                scores=scores_split[index],
                frame_meta=None if frame_meta_batch is None else frame_meta_batch.sample(index),
                gt_boxes=None if gt_boxes is None else gt_boxes[index],
                gt_box_labels=None if gt_box_labels is None else gt_box_labels[index],
            )
            for index in range(num_frames)
        )
    )


def concat_frame_ids(offset: torch.Tensor, point_to_batch: torch.Tensor) -> torch.Tensor:
    """Frame index per point from the batch ``offset`` (inclusive cumulative lengths).

    ``point_to_batch`` maps each point to its position in the batch-concatenated
    primary space: the point's own index for directly-concatenated points, or
    the ``inverse`` mapping for original-resolution points scattered from the
    sampled space.

    Args:
        offset: Inclusive cumulative frame lengths ``(B,)``.
        point_to_batch: Point to batch-concatenated position mapping.

    Returns:
        The frame index per point.
    """
    return torch.searchsorted(offset.to(point_to_batch.device), point_to_batch, right=True)
