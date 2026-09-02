"""Distributed (multi-rank) tests for the StreamPETR loss normalization.

These spawn real CPU process groups with the gloo backend so the actual
``reduce_mean_count`` collectives run.

The correctness property checked is the invariant that global (cross-rank)
normalization provides and rank-local normalization violates: **sharding a
batch across ranks must produce the same DDP-averaged loss as computing the
whole batch on one rank**. Rank-local counting fails this whenever ranks hold
different object counts (each GPU gets an equal vote regardless of load, an
upward-biased mean of per-rank means), which also made results silently depend
on the GPU count.

The DN test additionally pins the collective-uniformity contract: a rank whose
batch has no ground truth must still reach the DN count collective, otherwise
mixed-GT steps deadlock. Workers are joined with a timeout so a regression
shows up as a test failure rather than a hung pytest.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.task_modules.streaming import reduce_mean_count

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="gloo backend unavailable",
)

_WORLD_SIZE = 2
_NUM_CLASSES = 3
_NUM_QUERIES = 32


def _build_head() -> StreamPETRHead:
    torch.manual_seed(0)
    return StreamPETRHead(
        num_classes=_NUM_CLASSES,
        in_channels=128,
        hidden_dim=128,
        num_queries=_NUM_QUERIES,
        num_decoder_layers=3,
        num_heads=4,
        feedforward_channels=256,
        memory_len=32,
        topk_proposals=8,
        num_propagated=8,
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
            max_num=16,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=2.0),
            reg_cost=BBox3DL1Cost(
                weight=0.25, code_weights=(2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
            ),
            iou_cost=IoU3DCost(weight=0.0),
        ),
        point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
        code_weights=[2.0, 2.0] + [1.0] * 8,
    )


def _sample(
    num_boxes: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic (cls_scores, bbox_preds, gt_boxes, gt_labels) for one frame."""
    generator = torch.Generator().manual_seed(seed)
    cls_scores = torch.randn(_NUM_QUERIES, _NUM_CLASSES, generator=generator)
    bbox_preds = torch.randn(_NUM_QUERIES, 10, generator=generator)
    gt_boxes = torch.rand(num_boxes, 9, generator=generator) * 4.0 - 2.0
    gt_boxes[:, 3:6] = gt_boxes[:, 3:6].abs() + 1.0  # positive dims
    gt_labels = torch.arange(num_boxes, dtype=torch.long) % _NUM_CLASSES
    return cls_scores, bbox_preds, gt_boxes, gt_labels


def _spawn(worker, world_size: int = _WORLD_SIZE, timeout: float = 120.0) -> None:
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(
            worker,
            args=(world_size, os.path.join(directory, "store")),
            nprocs=world_size,
            join=False,
        )
        # A finite join converts a collective-uniformity regression (deadlock)
        # into a test failure instead of a hung pytest process. join(timeout)
        # returns after a single wait pass, so poll until the deadline.
        deadline = time.monotonic() + timeout
        while not context.join(timeout=5.0):
            if time.monotonic() > deadline:
                for process in context.processes:
                    process.terminate()
                raise AssertionError(f"distributed workers did not finish within {timeout}s")


def _reduce_mean_count_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        # Rank counts 2 and 8 (the section-6.4 worked example): mean must be 5.
        local_count = torch.tensor(float(2 if rank == 0 else 8))
        reduced = reduce_mean_count(local_count)
        assert reduced.item() == pytest.approx(5.0)
        # The input must not be mutated in place.
        assert local_count.item() == pytest.approx(2.0 if rank == 0 else 8.0)
    finally:
        dist.destroy_process_group()


def _sharding_invariance_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        head = _build_head().train()
        # Rank 0 holds the sparse frame (2 boxes), rank 1 the dense one (8).
        frames = [_sample(2, seed=10), _sample(8, seed=20)]

        # Sharded pass: each rank sees only its own frame.
        cls_scores, bbox_preds, gt_boxes, gt_labels = frames[rank]
        loss_cls, loss_bbox = head._loss_single(
            cls_scores.unsqueeze(0), bbox_preds.unsqueeze(0), [gt_boxes], [gt_labels]
        )
        sharded = torch.stack([loss_cls.detach(), loss_bbox.detach()])
        gathered = [torch.zeros_like(sharded) for _ in range(world_size)]
        dist.all_gather(gathered, sharded)
        ddp_averaged = torch.stack(gathered).mean(dim=0)

        # Combined pass: the same two frames as one batch on every rank.
        # reduce_mean_count then averages identical counts, so this equals the
        # true per-object mean over the union.
        combined_cls, combined_bbox = head._loss_single(
            torch.stack([frames[0][0], frames[1][0]]),
            torch.stack([frames[0][1], frames[1][1]]),
            [frames[0][2], frames[1][2]],
            [frames[0][3], frames[1][3]],
        )

        torch.testing.assert_close(ddp_averaged[0], combined_cls.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ddp_averaged[1], combined_bbox.detach(), rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


def _dn_mixed_gt_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        head = _build_head().train()
        if rank == 0:
            _, _, gt_boxes, gt_labels = _sample(4, seed=30)
        else:
            gt_boxes = torch.zeros(0, 9)
            gt_labels = torch.zeros(0, dtype=torch.long)

        # Minimal outputs contract for StreamPETRHead.loss: per-layer scores /
        # box params plus the DN bookkeeping. The GT-less rank gets
        # dn_mask_dict=None (what prepare_for_dn returns without boxes) and
        # must still reach the DN count collective inside loss().
        num_layers = head.num_decoder_layers
        generator = torch.Generator().manual_seed(40)
        all_cls_scores = torch.randn(num_layers, 1, _NUM_QUERIES, _NUM_CLASSES, generator=generator)
        all_bbox_preds = torch.randn(num_layers, 1, _NUM_QUERIES, 10, generator=generator)
        if rank == 0:
            pad = 8
            mask_dict = {
                "known_lbs_bboxes": (
                    gt_labels.repeat(head.scalar),
                    gt_boxes.repeat(head.scalar, 1),
                ),
                "known_indices": torch.arange(gt_labels.numel() * head.scalar),
                "batch_idx": torch.zeros(gt_labels.numel() * head.scalar, dtype=torch.long),
                "map_known_indice": torch.arange(gt_labels.numel() * head.scalar),
                "pad_size": pad,
                "output_known_lbs_bboxes": (
                    torch.randn(num_layers, 1, pad, _NUM_CLASSES, generator=generator),
                    torch.randn(num_layers, 1, pad, 10, generator=generator),
                ),
            }
        else:
            mask_dict = None
        outputs = {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "dn_mask_dict": mask_dict,
        }

        losses = head.loss(outputs, [gt_boxes], [gt_labels])

        # Both ranks completed (no deadlock) with a uniform key set.
        assert "dn_loss_cls" in losses and "dn_loss_bbox" in losses
        if rank == 1:
            assert losses["dn_loss_cls"].item() == 0.0
            assert losses["dn_loss_bbox"].item() == 0.0
    finally:
        dist.destroy_process_group()


def test_reduce_mean_count_is_identity_without_process_group() -> None:
    value = torch.tensor(3.0)
    assert reduce_mean_count(value) is value
    # Integer counts are cast to float either way, so the single-GPU dtype
    # matches the distributed path.
    assert reduce_mean_count(torch.tensor(3)).dtype == torch.float32


def test_reduce_mean_count_averages_counts_across_ranks() -> None:
    _spawn(_reduce_mean_count_worker)


def test_sharded_loss_matches_single_batch_equivalent() -> None:
    _spawn(_sharding_invariance_worker)


def test_dn_collective_survives_rank_without_ground_truth() -> None:
    _spawn(_dn_mixed_gt_worker)
