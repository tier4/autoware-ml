"""Tests for the shared ``seg_frames`` eval-output builder."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.metrics.segmentation3d.eval_output import (
    concat_frame_ids,
    segmentation_frames_eval_output,
)
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_detection_gt_batch,
)


def test_concat_frame_ids_buckets_by_offset() -> None:
    # Two frames with 3 and 2 sampled points -> offset [3, 5]. Original points
    # mapping (inverse) into sampled indices resolve to their frame.
    offset = torch.tensor([3, 5])
    inverse = torch.tensor([0, 2, 2, 3, 4, 4])
    assert concat_frame_ids(offset, inverse).tolist() == [0, 0, 0, 1, 1, 1]


def _batch_inputs(num_frames: int):
    """Build model inputs carrying the frame metadata and the boxes of every frame."""
    batch_inputs = build_batch_inputs(
        detection=build_detection_gt_batch(
            [torch.zeros((1, 9)), torch.zeros((0, 9))][:num_frames],
            [torch.tensor([0]), torch.zeros((0,), dtype=torch.long)][:num_frames],
        )
    )
    frame_meta = FrameMetaBatch(
        ego2globals=torch.eye(4).repeat(num_frames, 1, 1),
        scene_tokens=["scene-a", "scene-b"][:num_frames],
    )
    gt_batch = batch_inputs.multi_task_gt_batch._replace(frame_meta_batch=frame_meta)
    return batch_inputs.replace(multi_task_gt_batch=gt_batch)


def test_segmentation_frames_eval_output_splits_and_passes_meta() -> None:
    coord = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    pred = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 0, 0])
    scores = torch.full((4, 2), 0.5)
    frame_ids = torch.tensor([0, 0, 1, 1])
    batch_inputs = _batch_inputs(num_frames=2)
    out = segmentation_frames_eval_output(coord, pred, target, scores, frame_ids, 2, batch_inputs)
    frames = out["seg_frames"]
    assert len(frames) == 2
    assert frames[0]["pred"].tolist() == [0, 1]
    assert frames[1]["target"].tolist() == [0, 0]
    assert frames[0]["scene_token"] == "scene-a"
    assert frames[1]["scene_token"] == "scene-b"
    assert frames[0]["gt_boxes"].shape == (1, 9)
    assert frames[1]["gt_box_labels"].shape == (0,)


def test_segmentation_frames_eval_output_rejects_misaligned_meta() -> None:
    coord = torch.zeros((2, 3))
    labels = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="ego2global"):
        segmentation_frames_eval_output(
            coord,
            labels,
            labels,
            torch.full((2, 2), 0.5),
            torch.tensor([0, 1]),
            2,
            _batch_inputs(num_frames=1),
        )
