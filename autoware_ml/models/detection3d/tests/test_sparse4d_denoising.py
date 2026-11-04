"""Unit tests for the Sparse4D denoising training."""

from __future__ import annotations

from pathlib import Path

import torch

from autoware_ml.models.detection3d.heads.sparse4d import (
    ANCHOR_DIM,
    build_denoising_attention_mask,
    build_denoising_queries,
    encode_boxes_to_anchor_space,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)

_REG_WEIGHTS = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])
_NOISE_KWARGS = {
    "noise_center": 0.5,
    "noise_center_neg": 1.0,
    "noise_size": 0.2,
    "noise_yaw": 0.39,
    "noise_velocity": 1.0,
}


def _build_gt() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Two-sample batch: two boxes and one box (padding case)."""
    gt_boxes = [
        torch.tensor(
            [
                [2.5, 3.0, 0.2, 1.8, 0.9, 1.6, 0.1, 0.5, -0.2],
                [5.5, 5.0, -0.4, 3.6, 1.7, 1.5, -0.8, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[4.0, 2.0, 0.0, 1.0, 1.0, 1.8, 1.2, 0.1, 0.1]],
            dtype=torch.float32,
        ),
    ]
    gt_labels = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
    return gt_boxes, gt_labels


def _instance_ids(gt_boxes: list[torch.Tensor]) -> list[torch.Tensor]:
    """One distinct instance id per box."""
    return [torch.arange(boxes.shape[0]) for boxes in gt_boxes]


def _build_stage_inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse stage features for the fixture head (grid 8x8x4, 32 channels)."""
    generator = torch.Generator().manual_seed(0)
    voxels_per_sample = 6
    grid_coord = torch.stack(
        [
            torch.randint(0, 8, (batch_size * voxels_per_sample,), generator=generator),
            torch.randint(0, 8, (batch_size * voxels_per_sample,), generator=generator),
            torch.randint(0, 4, (batch_size * voxels_per_sample,), generator=generator),
        ],
        dim=1,
    )
    feat = torch.randn((batch_size * voxels_per_sample, 32), generator=generator)
    offset = torch.arange(1, batch_size + 1, dtype=torch.long) * voxels_per_sample
    return feat, grid_coord, offset


def test_denoising_queries_targets_and_layout() -> None:
    gt_boxes, gt_labels = _build_gt()
    torch.manual_seed(0)
    denoising = build_denoising_queries(
        gt_boxes,
        gt_labels,
        num_query=8,
        num_classes=2,
        query_budget=12,
        reg_weights=_REG_WEIGHTS,
        **_NOISE_KWARGS,
    )

    # budget 12 with max_gt 2 -> 3 groups of width 4 (2 positive + 2 negative).
    assert denoising.anchor.shape == (2, 12, ANCHOR_DIM)
    assert denoising.cls_targets.shape == (2, 12, 2)

    encoded_sample0 = encode_boxes_to_anchor_space(gt_boxes[0])
    for group in range(3):
        pos = slice(group * 4, group * 4 + 2)
        neg = slice(group * 4 + 2, group * 4 + 4)
        # Positives: clean targets, one-hot labels, full regression weights.
        assert torch.allclose(denoising.box_targets[0, pos], encoded_sample0)
        assert torch.allclose(denoising.box_weights[0, pos], _REG_WEIGHTS.expand(2, -1))
        assert torch.equal(denoising.cls_targets[0, pos].argmax(dim=-1), gt_labels[0])
        assert denoising.cls_targets[0, pos].sum() == 2.0
        # Negatives: background targets, no box supervision, cls supervised.
        assert denoising.cls_targets[0, neg].sum() == 0.0
        assert denoising.box_weights[0, neg].sum() == 0.0
        assert torch.all(denoising.cls_weights[0, group * 4 : group * 4 + 4] == 1.0)
        # Sample 1 has one box: its second slot per half is padding.
        assert denoising.cls_weights[1, group * 4 + 1] == 0.0
        assert denoising.cls_weights[1, group * 4 + 3] == 0.0
        assert denoising.box_weights[1, group * 4 + 1].sum() == 0.0

    # Noise bounds: z offsets stay within the configured half-extent fractions
    # and heading rotations within the yaw half-range.
    half_height = encoded_sample0[:, 5].exp() / 2.0
    for group in range(3):
        pos_anchor = denoising.anchor[0, group * 4 : group * 4 + 2]
        neg_anchor = denoising.anchor[0, group * 4 + 2 : group * 4 + 4]
        pos_dz = (pos_anchor[:, 2] - encoded_sample0[:, 2]).abs()
        neg_dz = (neg_anchor[:, 2] - encoded_sample0[:, 2]).abs()
        assert torch.all(pos_dz <= 0.5 * half_height + 1e-6)
        assert torch.all(neg_dz >= 0.5 * half_height - 1e-6)
        assert torch.all(neg_dz <= 1.0 * half_height + 1e-6)
        clean_yaw = torch.atan2(encoded_sample0[:, 6], encoded_sample0[:, 7])
        noised_yaw = torch.atan2(pos_anchor[:, 6], pos_anchor[:, 7])
        delta_yaw = torch.atan2(
            torch.sin(noised_yaw - clean_yaw), torch.cos(noised_yaw - clean_yaw)
        )
        assert torch.all(delta_yaw.abs() <= 0.39 + 1e-6)

    empty = build_denoising_queries(
        [torch.zeros((0, 9))],
        [torch.zeros((0,), dtype=torch.long)],
        num_query=8,
        num_classes=2,
        query_budget=12,
        reg_weights=_REG_WEIGHTS,
        **_NOISE_KWARGS,
    )
    assert empty is None


def test_denoising_attention_mask_structure() -> None:
    mask = build_denoising_attention_mask(4, 2, 3, torch.device("cpu"))

    assert mask.shape == (10, 10)
    # Regular queries never read the denoising tail but see each other.
    assert torch.all(mask[:4, 4:])
    assert not torch.any(mask[:4, :4])
    # Denoising groups read the regular queries and themselves only.
    assert not torch.any(mask[4:7, :4])
    assert not torch.any(mask[4:7, 4:7])
    assert torch.all(mask[4:7, 7:])
    assert torch.all(mask[7:, 4:7])
    assert not torch.any(mask[7:, 7:])


def test_denoising_isolation_keeps_real_outputs_identical(tmp_path: Path) -> None:
    head = build_sparse4d_head(
        make_anchor_file(tmp_path), dn_query_budget=12, dropout=0.0, attn_drop=0.0
    )
    inputs = _build_stage_inputs()
    gt_boxes, gt_labels = _build_gt()

    with torch.no_grad():
        plain = head.eval()([inputs], gt_boxes=gt_boxes, gt_labels=gt_labels)
        torch.manual_seed(0)
        denoised = head.train()([inputs], gt_boxes=gt_boxes, gt_labels=gt_labels)

    # The eval step runs the fused attention kernel, hence the float tolerance
    assert plain.denoising is None
    assert torch.allclose(plain.classification, denoised.classification, atol=1e-5)
    assert torch.allclose(plain.prediction, denoised.prediction, atol=1e-5)
    assert torch.allclose(plain.quality, denoised.quality, atol=1e-5)
    assert denoised.denoising.classification.shape == (2, 2, 12, 2)
    assert denoised.denoising.prediction.shape == (2, 2, 12, ANCHOR_DIM)


def test_denoising_loss_terms(tmp_path: Path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path), dn_query_budget=12)
    inputs = _build_stage_inputs()
    gt_boxes, gt_labels = _build_gt()

    torch.manual_seed(0)
    outputs = head([inputs], gt_boxes=gt_boxes, gt_labels=gt_labels)
    losses = head.loss(outputs, gt_boxes, gt_labels, _instance_ids(gt_boxes), [0, 1])

    for prefix in ("layer_0", "layer_-1"):
        assert torch.isfinite(losses[f"{prefix}_loss_dn_cls"])
        assert torch.isfinite(losses[f"{prefix}_loss_dn_bbox"])
        assert losses[f"{prefix}_loss_dn_bbox"] > 0.0
    # 3 groups x 3 boxes in the batch = 9 positive denoising queries.
    assert losses["dn_num_pos"] == 9.0
    dn_sum = sum(value for key, value in losses.items() if "loss_dn" in key)
    assert losses["loss"] > dn_sum

    losses.pop("loss").backward()
    assert head.instance_bank.anchor.grad is not None


def test_denoising_skipped_without_boxes(tmp_path: Path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path), dn_query_budget=12)
    inputs = _build_stage_inputs()
    gt_boxes = [torch.zeros((0, 9)), torch.zeros((0, 9))]
    gt_labels = [torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)]

    outputs = head([inputs], gt_boxes=gt_boxes, gt_labels=gt_labels)
    losses = head.loss(outputs, gt_boxes, gt_labels, _instance_ids(gt_boxes), [0, 1])

    assert outputs.denoising is None
    assert "dn_num_pos" not in losses
    assert torch.isfinite(losses["loss"])


def test_denoising_runs_with_temporal_memory(tmp_path: Path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path), num_temp_instances=2, dn_query_budget=12)
    inputs = _build_stage_inputs()
    gt_boxes, gt_labels = _build_gt()
    pose = torch.eye(4, dtype=torch.float64)[None].repeat(2, 1, 1)

    for frame_index in range(2):
        outputs = head(
            [inputs],
            timestamps_seconds=torch.full((2,), 10.0 + 0.1 * frame_index, dtype=torch.float64),
            lidar2global=pose,
            scene_tokens=["scene_a", "scene_b"],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_instance_ids=_instance_ids(gt_boxes),
            gt_num_points=[torch.full((boxes.shape[0],), 5) for boxes in gt_boxes],
        )
        losses = head.loss(outputs, gt_boxes, gt_labels, _instance_ids(gt_boxes), [0, 1])
        assert outputs.denoising is not None
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()

    # The temporal state holds regular queries only, never the denoising tail
    assert head.instance_bank._cached_feature.shape == (2, 2, 32)


def test_denoising_masks_unknown_velocity() -> None:
    gt_boxes, gt_labels = _build_gt()
    gt_boxes[1][0, 7:9] = float("nan")
    denoising = build_denoising_queries(
        gt_boxes,
        gt_labels,
        num_query=8,
        num_classes=2,
        query_budget=12,
        reg_weights=_REG_WEIGHTS,
        **_NOISE_KWARGS,
    )

    assert torch.isfinite(denoising.anchor).all()
    assert torch.isfinite(denoising.box_targets).all()
    assert torch.all(denoising.box_weights[1, 0, 8:] == 0.0)
    assert torch.all(denoising.box_weights[1, 0, :8] == _REG_WEIGHTS[:8])
