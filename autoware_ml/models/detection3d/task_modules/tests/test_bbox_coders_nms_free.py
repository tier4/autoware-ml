"""Unit tests for the NMS-free 3D box coder used by query-based detectors."""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    NMSFreeBBoxCoder3D,
    denormalize_boxes3d,
    normalize_boxes3d,
)


def _metric_boxes() -> torch.Tensor:
    # [cx, cy, cz, dx, dy, dz, yaw, vx, vy]
    return torch.tensor(
        [
            [1.0, 2.0, -0.5, 4.0, 2.0, 1.5, 0.3, 1.0, -2.0],
            [-3.0, 0.5, 0.25, 0.8, 0.8, 1.7, -2.9, 0.0, 0.0],
        ]
    )


def test_normalize_denormalize_round_trip() -> None:
    boxes = _metric_boxes()

    restored = denormalize_boxes3d(normalize_boxes3d(boxes))

    torch.testing.assert_close(restored, boxes, rtol=1e-5, atol=1e-6)


def test_normalize_uses_log_sizes_and_split_yaw() -> None:
    boxes = _metric_boxes()

    encoded = normalize_boxes3d(boxes)

    assert encoded.shape == (2, 10)
    torch.testing.assert_close(encoded[:, 3:6], boxes[:, 3:6].log())
    torch.testing.assert_close(encoded[:, 6], boxes[:, 6].sin())
    torch.testing.assert_close(encoded[:, 7], boxes[:, 6].cos())


def test_decode_maps_flat_topk_index_to_class_and_box() -> None:
    coder = NMSFreeBBoxCoder3D(pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], max_num=2)
    # Two queries, three classes; the strongest scores are query1/class2 then
    # query0/class1, so labels and boxes must follow that pairing.
    logits = torch.tensor([[-9.0, 1.0, -9.0], [-9.0, -9.0, 5.0]])
    encoded = normalize_boxes3d(_metric_boxes())

    predictions = coder.decode(logits.unsqueeze(0), encoded.unsqueeze(0))[0]

    assert predictions["labels"].tolist() == [2, 1]
    torch.testing.assert_close(predictions["scores"], logits.sigmoid().flatten().topk(2).values)
    torch.testing.assert_close(
        predictions["bboxes"], denormalize_boxes3d(encoded[[1, 0]]), rtol=1e-5, atol=1e-6
    )


def test_decode_applies_score_threshold_and_post_center_range() -> None:
    encoded = normalize_boxes3d(_metric_boxes())
    logits = torch.tensor([[2.0, -9.0, -9.0], [-9.0, -9.0, -1.0]])

    thresholded = NMSFreeBBoxCoder3D(
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], max_num=6, score_threshold=0.5
    ).decode(logits.unsqueeze(0), encoded.unsqueeze(0))[0]
    assert thresholded["labels"].numel() == 1
    assert (thresholded["scores"] >= 0.5).all()

    # Box 0 sits at x=1, box 1 at x=-3; a range that keeps only positive x
    # must drop box 1 together with its score and label.
    ranged = NMSFreeBBoxCoder3D(
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        post_center_range=[0.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        max_num=6,
    ).decode(logits.unsqueeze(0), encoded.unsqueeze(0))[0]
    assert ranged["bboxes"].shape[0] == ranged["scores"].numel() == ranged["labels"].numel()
    assert (ranged["bboxes"][:, 0] >= 0.0).all()


def test_decode_returns_one_entry_per_sample() -> None:
    coder = NMSFreeBBoxCoder3D(pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], max_num=1)
    encoded = normalize_boxes3d(_metric_boxes()).unsqueeze(0).repeat(3, 1, 1)
    logits = torch.zeros(3, 2, 3)

    predictions = coder.decode(logits, encoded)

    assert len(predictions) == 3
    assert all(p["bboxes"].shape[0] == 1 for p in predictions)
