"""Tests for the shared ``seg_frames`` eval-output builder."""

from __future__ import annotations

import numpy as np
import torch

from autoware_ml.metrics.segmentation3d.eval_output import (
    concat_frame_ids,
    segmentation_frames_eval_output,
)


def test_concat_frame_ids_buckets_by_offset() -> None:
    # Two frames with 3 and 2 sampled points -> offset [3, 5]. Original points
    # mapping (inverse) into sampled indices resolve to their frame.
    offset = torch.tensor([3, 5])
    inverse = torch.tensor([0, 2, 2, 3, 4, 4])
    assert concat_frame_ids(offset, inverse).tolist() == [0, 0, 0, 1, 1, 1]


def test_segmentation_frames_eval_output_splits_and_passes_meta() -> None:
    coord = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    pred = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 0, 0])
    scores = torch.full((4, 2), 0.5)
    frame_ids = torch.tensor([0, 0, 1, 1])
    batch = {
        "ego2global": [np.eye(4), np.eye(4)],
        "scene_token": ["scene-a", "scene-b"],
        "gt_boxes": [torch.zeros((1, 9)), torch.zeros((0, 9))],
        "gt_labels": [torch.tensor([0]), torch.zeros((0,), dtype=torch.long)],
    }
    out = segmentation_frames_eval_output(coord, pred, target, scores, frame_ids, 2, batch)
    frames = out["seg_frames"]
    assert len(frames) == 2
    assert frames[0]["pred"].tolist() == [0, 1]
    assert frames[1]["target"].tolist() == [0, 0]
    assert frames[0]["scene_token"] == "scene-a"
    assert frames[1]["scene_token"] == "scene-b"
    assert frames[0]["gt_boxes"].shape == (1, 9)
    assert frames[1]["gt_box_labels"].shape == (0,)
