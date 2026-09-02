"""Tests for StreamPETR partial-ignore and the auxiliary 2D head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.partial_ignore import (
    mask_ignored_columns,
    normalize_status_flags,
    resolve_partial_ignore_labels,
)
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]


def _build_head(partial_ignore: bool) -> StreamPETRHead:
    return StreamPETRHead(
        num_classes=7,
        in_channels=32,
        hidden_dim=32,
        num_queries=16,
        num_decoder_layers=2,
        num_heads=4,
        feedforward_channels=64,
        memory_len=16,
        topk_proposals=4,
        num_propagated=4,
        with_dn=True,
        with_ego_pos=True,
        depth_num=8,
        LID=True,
        position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
        scalar=2,
        noise_scale=0.5,
        dn_weight=1.0,
        split=0.5,
        use_bottom_center=True,
        bbox_coder=NMSFreeBBoxCoder3D(
            pc_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
            post_center_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            score_threshold=0.01,
            max_num=8,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=2.0),
            reg_cost=BBox3DL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.0),
        ),
        point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
        code_weights=[2.0, 2.0] + [1.0] * 8,
        class_names=CLASS_NAMES if partial_ignore else None,
        partial_ignore_classes=["traffic_cone", "barrier"] if partial_ignore else None,
    )


def test_resolve_partial_ignore_labels() -> None:
    assert resolve_partial_ignore_labels(CLASS_NAMES, ["traffic_cone", "barrier"]) == [5, 6]
    assert resolve_partial_ignore_labels(CLASS_NAMES, None) is None


def test_normalize_status_flags_handles_tensors_and_lists() -> None:
    assert normalize_status_flags([True, False], 2) == [True, False]
    assert normalize_status_flags(torch.tensor([1.0, 0.0]), 2) == [True, False]
    assert normalize_status_flags([torch.tensor(False), torch.tensor(True)], 2) == [False, True]


def test_normalize_status_flags_rejects_missing_and_mismatched_flags() -> None:
    with pytest.raises(ValueError, match="annotation_status is required"):
        normalize_status_flags(None, 2)
    with pytest.raises(ValueError, match="2 flags for a batch of 3"):
        normalize_status_flags([True, False], 3)


def test_mask_ignored_columns_zeroes_selected_rows_only() -> None:
    weights = torch.ones(4, 7)
    mask_ignored_columns(weights, torch.tensor([1, 3]), [5, 6])
    assert torch.all(weights[[1, 3]][:, [5, 6]] == 0.0)
    assert torch.all(weights[[0, 2]] == 1.0)
    assert torch.all(weights[:, :5] == 1.0)
    # No rows selected: a no-op.
    mask_ignored_columns(weights, torch.zeros(0, dtype=torch.long), [5, 6])


def test_sigmoid_focal_loss_classwise_weights_zero_masked_columns() -> None:
    loss_fn = SigmoidFocalLoss()
    logits = torch.randn(4, 7)
    targets = torch.zeros(4, 7)
    ones = torch.ones(4, 7)
    baseline = loss_fn(logits, targets, avg_factor=1.0)
    weighted = loss_fn(logits, targets, weights=ones, avg_factor=1.0)
    assert torch.allclose(baseline, weighted)

    masked_weights = ones.clone()
    masked_weights[:, [5, 6]] = 0.0
    masked = loss_fn(logits, targets, weights=masked_weights, avg_factor=1.0)
    manual = loss_fn(logits[:, :5], targets[:, :5], avg_factor=1.0)
    assert torch.allclose(masked, manual, atol=1e-6)


def test_get_targets_zeroes_ignore_columns_on_every_query_of_unannotated_frames() -> None:
    head = _build_head(partial_ignore=True)
    num_queries = 16
    cls_logits = torch.randn(1, num_queries, 7)
    box_params = torch.randn(1, num_queries, 10)
    box_params[..., :3] = 0.0
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    targets = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels, [False])[0]
    assert targets.label_weights is not None
    assert targets.label_weights.shape == (num_queries, 7)

    # Every query of the un-annotated frame loses only the cone/barrier columns.
    assert torch.all(targets.label_weights[:, [5, 6]] == 0.0)
    assert torch.all(targets.label_weights[:, :5] == 1.0)

    # A fully annotated batch skips the classwise-weight tensors entirely.
    annotated = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels, [True])[0]
    assert annotated.label_weights is None

    # In a mixed batch every sample gets a weights tensor; the annotated one
    # stays all-ones so the stacked tensor is uniform.
    mixed = head._get_targets(
        cls_logits.repeat(2, 1, 1),
        box_params.repeat(2, 1, 1),
        gt_boxes * 2,
        gt_labels * 2,
        [True, False],
    )
    assert torch.all(mixed[0].label_weights == 1.0)
    assert torch.all(mixed[1].label_weights[:, [5, 6]] == 0.0)


def test_dn_label_weights_mask_all_rows_of_unannotated_samples() -> None:
    head = _build_head(partial_ignore=True)
    cls_scores = torch.randn(6, 7)
    known_labels = torch.tensor([0, 7, 7, 1, 7, 5])
    known_bids = torch.tensor([0, 0, 1, 1, 1, 0])
    weights = head._dn_label_weights(cls_scores, known_labels, known_bids, [True, False])
    assert weights is not None
    # Every row of sample 1 (indices 2, 3, 4) is masked on columns 5/6.
    assert torch.all(weights[[2, 3, 4]][:, [5, 6]] == 0.0)
    assert torch.all(weights[[2, 3, 4]][:, :5] == 1.0)
    # Rows of the annotated sample 0 stay fully weighted.
    assert torch.all(weights[[0, 1, 5]] == 1.0)

    assert head._dn_label_weights(cls_scores, known_labels, known_bids, [True, True]) is None


def test_head_loss_wiring_applies_annotation_status() -> None:
    """head.loss must thread the status flags through to the focal loss."""
    torch.manual_seed(0)
    head = _build_head(partial_ignore=True).eval()
    outputs = {
        "all_cls_scores": torch.randn(2, 1, 16, 7),
        "all_bbox_preds": torch.rand(2, 1, 16, 10),
        "dn_mask_dict": None,
    }
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    annotated = head.loss(outputs, gt_boxes, gt_labels, annotation_status=[True])
    ignored = head.loss(outputs, gt_boxes, gt_labels, annotation_status=[False])
    # Masking removes non-negative background terms from the cls loss only.
    assert ignored["loss_cls"] < annotated["loss_cls"]
    assert torch.allclose(ignored["loss_bbox"], annotated["loss_bbox"])

    with pytest.raises(ValueError, match="annotation_status is required"):
        head.loss(outputs, gt_boxes, gt_labels)

    # Heads without partial-ignore never require the flags.
    plain_head = _build_head(partial_ignore=False).eval()
    plain_losses = plain_head.loss(outputs, gt_boxes, gt_labels)
    assert torch.isfinite(plain_losses["loss"])


def test_focal_head_2d_forward_and_loss_with_partial_ignore() -> None:
    torch.manual_seed(0)
    head = FocalHead2D(
        num_classes=7,
        in_channels=32,
        embed_dims=32,
        stride=16,
        class_names=CLASS_NAMES,
        partial_ignore_classes=["traffic_cone", "barrier"],
    )
    batch_size, num_cams = 2, 2
    img_features = torch.randn(batch_size, num_cams, 32, 6, 10)
    outputs = head(img_features, image_height=96, image_width=160)
    assert outputs["enc_cls_scores"].shape == (4, 60, 7)
    assert outputs["enc_bbox_preds"].shape == (4, 60, 4)
    assert outputs["pred_centers2d"].shape == (4, 60, 2)
    assert outputs["centerness"].shape == (4, 60, 1)

    gt_boxes = np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32)
    gt_centers = np.array([[35.0, 35.0]], dtype=np.float32)
    gt_labels = np.array([5], dtype=np.int64)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_centers = np.zeros((0, 2), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int64)
    annotations = {
        "gt_bboxes_2d": [[gt_boxes, empty_boxes], [empty_boxes, empty_boxes]],
        "gt_labels_2d": [[gt_labels, empty_labels], [empty_labels, empty_labels]],
        "centers_2d": [[gt_centers, empty_centers], [empty_centers, empty_centers]],
    }

    losses = head.loss(outputs, annotation_status=[True, False], **annotations)
    for key in (
        "loss_cls2d",
        "loss_bbox2d",
        "loss_iou2d",
        "loss_centers2d",
        "loss_centerness2d",
    ):
        assert torch.isfinite(losses[key]), key

    # Masking removes strictly positive background terms of sample 1's images
    # and touches nothing else.
    unmasked = head.loss(outputs, annotation_status=[True, True], **annotations)
    assert losses["loss_cls2d"] < unmasked["loss_cls2d"]
    for key in ("loss_bbox2d", "loss_iou2d", "loss_centers2d", "loss_centerness2d"):
        assert torch.allclose(losses[key], unmasked[key]), key

    with pytest.raises(ValueError, match="annotation_status is required"):
        head.loss(outputs, **annotations)


def test_focal_head_2d_partial_ignore_class_weights_select_exact_rows() -> None:
    head = FocalHead2D(
        num_classes=7,
        in_channels=32,
        embed_dims=32,
        class_names=CLASS_NAMES,
        partial_ignore_classes=["traffic_cone", "barrier"],
    )
    tokens_per_image = 3
    flat_scores = torch.randn(4 * tokens_per_image, 7)
    weights = head._partial_ignore_class_weights(
        flat_scores, [True, False, False, True], tokens_per_image
    )
    assert weights is not None
    expected = torch.ones_like(flat_scores)
    expected[3:9, 5:7] = 0.0
    assert torch.equal(weights, expected)

    assert head._partial_ignore_class_weights(flat_scores, [True] * 4, tokens_per_image) is None
