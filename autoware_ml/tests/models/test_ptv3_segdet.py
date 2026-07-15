"""Unit tests for the PTv3 joint segdet model's supervision masking and fusion."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from autoware_ml.models.detection3d.ptv3 import PTv3DetFeatureFusion
from autoware_ml.models.multi.ptv3_segdet import PTv3SegDetModel
from autoware_ml.utils.point_cloud.structures import Point


def _make_masking_model(recorded_calls: list) -> SimpleNamespace:
    """Build a duck-typed segdet model recording bbox_head.loss inputs."""

    def bbox_loss(det_outputs, gt_boxes, gt_labels):
        recorded_calls.append((det_outputs, gt_boxes, gt_labels))
        return {"loss": det_outputs["heatmap"].sum() * 0.0 + 1.0}

    def seg_loss(seg_logits, segment):
        zero = seg_logits.sum() * 0.0
        return {"loss_ce": zero, "loss_lovasz": zero, "loss": zero}

    return SimpleNamespace(
        seg3d_head=SimpleNamespace(loss=seg_loss),
        bbox_head=SimpleNamespace(loss=bbox_loss),
        segmentation_loss_weight=1.0,
        detection_loss_weight=1.0,
        _detection_frame_mask=PTv3SegDetModel._detection_frame_mask,
        _mask_detection_outputs=PTv3SegDetModel._mask_detection_outputs,
        _mask_list=PTv3SegDetModel._mask_list,
    )


def _make_batch(has_boxes: list[bool]) -> dict:
    """Detection supervision is carried by the ground truth itself: frames
    without boxes are detection-unsupervised."""
    return {
        "segment": torch.tensor([0, 1], dtype=torch.long),
        "gt_boxes": [
            torch.full((1, 9), float(i)) if with_boxes else torch.zeros((0, 9))
            for i, with_boxes in enumerate(has_boxes)
        ],
        "gt_labels": [
            torch.tensor([i]) if with_boxes else torch.zeros((0,), dtype=torch.long)
            for i, with_boxes in enumerate(has_boxes)
        ],
    }


def _make_outputs(batch_size: int) -> dict:
    return {
        "seg_logits": torch.randn(4, 3, requires_grad=True),
        "det_outputs": {
            "heatmap": torch.randn(batch_size, 2, 8, requires_grad=True),
            "center": torch.randn(batch_size, 2, 8, requires_grad=True),
        },
    }


def test_compute_metrics_masks_detection_loss_to_frames_with_boxes() -> None:
    recorded_calls: list = []
    model = _make_masking_model(recorded_calls)
    batch = _make_batch([True, False])
    outputs = _make_outputs(batch_size=2)

    metrics = PTv3SegDetModel.compute_metrics(model, batch, outputs)

    assert len(recorded_calls) == 1
    det_outputs, gt_boxes, gt_labels = recorded_calls[0]
    assert det_outputs["heatmap"].shape[0] == 1
    assert torch.equal(det_outputs["heatmap"][0], outputs["det_outputs"]["heatmap"][0])
    assert len(gt_boxes) == 1 and float(gt_boxes[0][0, 0]) == 0.0
    assert len(gt_labels) == 1
    assert "det_loss" in metrics and "loss" in metrics


def test_compute_metrics_keeps_every_frame_when_all_have_boxes() -> None:
    recorded_calls: list = []
    model = _make_masking_model(recorded_calls)
    batch = _make_batch([True, True])
    outputs = _make_outputs(batch_size=2)

    PTv3SegDetModel.compute_metrics(model, batch, outputs)

    det_outputs, gt_boxes, _ = recorded_calls[0]
    assert torch.equal(det_outputs["heatmap"], outputs["det_outputs"]["heatmap"])
    assert len(gt_boxes) == 2


def test_compute_metrics_keeps_det_branch_in_graph_without_supervised_frames() -> None:
    recorded_calls: list = []
    model = _make_masking_model(recorded_calls)
    batch = _make_batch([False, False])
    outputs = _make_outputs(batch_size=2)

    metrics = PTv3SegDetModel.compute_metrics(model, batch, outputs)

    assert not recorded_calls
    assert float(metrics["det_loss"].detach()) == 0.0
    # The zero detection loss must stay connected to the detection outputs so
    # DDP reducers see gradients (of zero) for every detection parameter.
    assert metrics["det_loss"].grad_fn is not None
    assert float(metrics["loss"].detach()) == 0.0


def _make_eval_model() -> SimpleNamespace:
    def predict(det_outputs):
        batch_size = det_outputs["heatmap"].shape[0]
        return [
            {
                "bboxes_3d": torch.full((2, 9), float(index)),
                "scores_3d": torch.full((2,), float(index)),
                "labels_3d": torch.zeros(2, dtype=torch.long),
            }
            for index in range(batch_size)
        ]

    return SimpleNamespace(
        bbox_head=SimpleNamespace(predict=predict),
        _detection_frame_mask=PTv3SegDetModel._detection_frame_mask,
    )


def test_build_eval_output_neutralizes_unflagged_frames_keeping_one_entry_per_frame() -> None:
    """Per-frame metric-state alignment: torchmetrics list-state sync issues one
    all_gather per entry, so every rank must contribute exactly one entry per
    frame regardless of its seg/det frame mix."""
    model = _make_eval_model()
    outputs = _make_outputs(batch_size=2)
    batch = {
        **_make_batch([False, True]),
        "inverse": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "origin_segment": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "origin_coord": torch.zeros((4, 3)),
    }

    eval_out = PTv3SegDetModel.build_eval_output(model, batch, outputs)

    assert len(eval_out["predictions"]) == 2
    assert len(eval_out["gt_boxes"]) == 2
    # Unflagged frame: empty predictions with preserved trailing dims.
    assert eval_out["predictions"][0]["bboxes_3d"].shape == (0, 9)
    assert eval_out["predictions"][0]["scores_3d"].shape == (0,)
    # Flagged frame: predictions kept as-is.
    assert eval_out["predictions"][1]["bboxes_3d"].shape == (2, 9)
    assert float(eval_out["predictions"][1]["scores_3d"][0]) == 1.0


def test_build_eval_output_without_flagged_frames_keeps_neutral_entries() -> None:
    model = _make_eval_model()
    outputs = _make_outputs(batch_size=1)
    batch = {
        **_make_batch([False]),
        "inverse": torch.tensor([0, 1], dtype=torch.long),
        "origin_segment": torch.tensor([0, 1], dtype=torch.long),
        "origin_coord": torch.zeros((2, 3)),
    }

    eval_out = PTv3SegDetModel.build_eval_output(model, batch, outputs)

    assert len(eval_out["predictions"]) == 1
    assert eval_out["predictions"][0]["bboxes_3d"].shape == (0, 9)
    assert len(eval_out["gt_boxes"]) == 1


def test_seg_head_loss_returns_connected_zero_when_all_targets_ignored() -> None:
    """A batch can carry zero seg supervision (e.g. seg-masked det-val frames);
    CE over zero valid targets is nan, so the head must short-circuit to a
    graph-connected zero."""
    from autoware_ml.tests.models.ptv3_detection_fixtures import build_seg_head

    head = build_seg_head(num_classes=3, dec_depths=(0,))
    logits = torch.randn(5, 3, requires_grad=True)
    all_ignored = torch.full((5,), -1, dtype=torch.long)

    metrics = head.loss(logits, all_ignored)

    assert float(metrics["loss_ce"].detach()) == 0.0
    assert float(metrics["loss"].detach()) == 0.0
    assert metrics["loss"].grad_fn is not None
    # Sanity: valid targets still produce a real loss.
    assert torch.isfinite(head.loss(logits, torch.tensor([0, 1, 2, 0, 1]))["loss"])


def test_det_feature_fusion_reads_pooling_chain_non_destructively() -> None:
    torch.manual_seed(0)
    fusion = PTv3DetFeatureFusion(in_channels=16, skip_channels=8, out_channels=16).eval()
    parent_feat = torch.randn(6, 8)
    parent = Point(
        feat=parent_feat.clone(),
        grid_coord=torch.randint(0, 8, (6, 3)),
        offset=torch.tensor([6], dtype=torch.long),
    )
    deepest = Point(
        feat=torch.randn(3, 16),
        pooling_parent=parent,
        pooling_inverse=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
    )

    fused, grid_coord, offset = fusion(deepest)

    assert fused.shape == (6, 16)
    assert torch.equal(grid_coord, parent.grid_coord)
    assert torch.equal(offset, parent.offset)
    # Nothing popped, nothing mutated: the segmentation decoder still sees the
    # intact chain afterwards.
    assert "pooling_parent" in deepest
    assert "pooling_inverse" in deepest
    assert torch.equal(parent.feat, parent_feat)
