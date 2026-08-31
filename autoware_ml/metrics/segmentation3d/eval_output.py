"""Shared eval-output builder for the 3D segmentation suites.

Both segmentation suites consume one contract: ``seg_frames``, a list with one
entry per frame carrying that frame's points (``coord``, predicted/target labels,
per-class ``scores``) plus any per-frame metadata the configured filters need
(``ego2global``, ``scene_token``) and, for the cross-task partial detection
metric, the frame's detection GT boxes.

Metadata keys are copied from the batch when the dataset supplies them. A filter
or metric that needs a missing key fails loud in the suite with a message naming
it, so absence is never silent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

# Per-frame metadata passed through to each frame entry when the batch carries it.
_FRAME_META_KEYS = ("ego2global", "scene_token", "timestamp")
# Per-frame detection ground truth for the cross-task partial-detection metric,
# batch key to the key it takes inside a frame entry.
_FRAME_BOX_KEYS = {"gt_boxes": "gt_boxes", "gt_labels": "gt_box_labels"}


def segmentation_frames_eval_output(
    coord: torch.Tensor,
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    scores: torch.Tensor,
    frame_ids: torch.Tensor,
    num_frames: int,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Split batch-concatenated per-point tensors into the ``seg_frames`` list.

    Args:
        coord: ``(N, 3+)`` point coordinates (base_link).
        pred_labels: ``(N,)`` predicted class per point.
        target_labels: ``(N,)`` ground-truth class per point.
        scores: ``(N, C)`` per-class scores per point.
        frame_ids: ``(N,)`` frame index of every point.
        num_frames: Number of frames in the batch.
        batch: Batch dictionary, per-frame metadata and detection GT are copied
            into each frame entry when present.

    Returns:
        ``{"seg_frames": [...]}`` with one entry per frame.
    """
    for key in (*_FRAME_META_KEYS, *_FRAME_BOX_KEYS):
        if key in batch and len(batch[key]) != num_frames:
            raise ValueError(
                f"batch[{key!r}] has {len(batch[key])} entries for {num_frames} frames, "
                "per-frame metadata must align one-to-one or points get another frame's context."
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

    frames: list[dict[str, Any]] = []
    for frame_index in range(num_frames):
        frame: dict[str, Any] = {
            "coord": coord_split[frame_index],
            "pred": pred_split[frame_index],
            "target": target_split[frame_index],
            "scores": scores_split[frame_index],
        }
        for key in _FRAME_META_KEYS:
            if key in batch:
                frame[key] = batch[key][frame_index]
        for batch_key, frame_key in _FRAME_BOX_KEYS.items():
            if batch_key in batch:
                frame[frame_key] = batch[batch_key][frame_index]
        frames.append(frame)
    return {"seg_frames": frames}


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
