# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import copy
import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    StreamPETRDenoisingOutputs,
    StreamPETRHeadOutputs,
)
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
_CLASS_NAMES = ["car", "truck", "bus"]
_NUM_CLASSES = len(_CLASS_NAMES)
_NUM_QUERIES = 32
# _build_head's decoder layers and denoising groups.
_NUM_LAYERS = 3
_SCALAR = 2
_FEATURES = 16


def _build_head() -> StreamPETRHead:
    torch.manual_seed(0)
    return StreamPETRHead(
        class_names=_CLASS_NAMES,
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
            point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
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


def _spawn(worker, *args, world_size: int = _WORLD_SIZE, timeout: float = 120.0) -> None:
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(
            worker,
            args=(world_size, os.path.join(directory, "store"), *args),
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
        # Rank counts 2 and 8: the mean is 5.
        local_count = torch.tensor(float(2 if rank == 0 else 8))
        reduced = reduce_mean_count(local_count)
        assert reduced.item() == pytest.approx(5.0)
        # The input must not be mutated in place.
        assert local_count.item() == pytest.approx(2.0 if rank == 0 else 8.0)
        # The floor applies to the count of all ranks: 1 and 0 positives are 1 over 2 ranks,
        # 0 and 0 are floored to 1 over 2 ranks; the mean itself is not floored.
        assert reduce_mean_count(torch.tensor(float(1 - rank))).item() == pytest.approx(0.5)
        assert reduce_mean_count(torch.tensor(0.0)).item() == pytest.approx(0.5)
        # Several counts reduce in one collective, floored one by one.
        pair = reduce_mean_count(torch.tensor([float(1 - rank), 4.0]))
        assert pair.tolist() == pytest.approx([0.5, 4.0])
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
            cls_scores.unsqueeze(0),
            bbox_preds.unsqueeze(0),
            [gt_boxes],
            [gt_labels],
            label_weights=torch.ones(1, 1, 1),
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
            label_weights=torch.ones(2, 1, 1),
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
        # box params plus the denoising queries. The GT-less rank gets
        # denoising=None (what the head returns without boxes) and must
        # still reach the DN count collective inside loss().
        num_layers = head.num_decoder_layers
        generator = torch.Generator().manual_seed(40)
        all_cls_scores = torch.randn(num_layers, 1, _NUM_QUERIES, _NUM_CLASSES, generator=generator)
        all_bbox_preds = torch.randn(num_layers, 1, _NUM_QUERIES, 10, generator=generator)
        if rank == 0:
            pad = 8
            num_known = gt_labels.numel() * head.scalar
            denoising = StreamPETRDenoisingOutputs(
                cls_scores=torch.randn(num_layers, 1, pad, _NUM_CLASSES, generator=generator),
                bbox_preds=torch.randn(num_layers, 1, pad, 10, generator=generator),
                known_labels=gt_labels.repeat(head.scalar),
                known_boxes=gt_boxes.repeat(head.scalar, 1),
                batch_indices=torch.zeros(num_known, dtype=torch.long),
                slot_indices=torch.arange(num_known),
            )
        else:
            denoising = None
        outputs = StreamPETRHeadOutputs(
            all_cls_scores=all_cls_scores, all_bbox_preds=all_bbox_preds, denoising=denoising
        )

        losses = head.loss(
            outputs, [gt_boxes], [gt_labels], padding_mask=torch.zeros(1, dtype=torch.bool)
        )

        # Both ranks completed (no deadlock) with a uniform key set.
        assert "dn_loss_cls" in losses and "dn_loss_bbox" in losses
        if rank == 1:
            assert losses["dn_loss_cls"].item() == 0.0
            assert losses["dn_loss_bbox"].item() == 0.0
    finally:
        dist.destroy_process_group()


def _frame(
    num_boxes: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query and denoising features of one frame with its boxes and labels.

    The denoising features hold ``_SCALAR`` queries per box, in the order of
    ``labels.repeat(_SCALAR)``.
    """
    generator = torch.Generator().manual_seed(seed)
    queries = torch.randn(_NUM_LAYERS, _NUM_QUERIES, _FEATURES, generator=generator)
    known = torch.randn(_NUM_LAYERS, _SCALAR * num_boxes, _FEATURES, generator=generator)
    _, _, gt_boxes, gt_labels = _sample(num_boxes, seed)
    return queries, known, gt_boxes, gt_labels


def _layer_loss(
    head: StreamPETRHead,
    layer: torch.nn.Module,
    frames: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Total head loss of the frames, every logit and box made by one call of ``layer``."""
    sizes = [queries.shape[1] + known.shape[1] for queries, known, _, _ in frames]
    num_known = [known.shape[1] for _, known, _, _ in frames]
    outputs = layer(torch.cat([torch.cat(frame[:2], dim=1) for frame in frames], dim=1))
    per_frame = outputs.split(sizes, dim=1)
    known_outputs = torch.stack(
        [
            torch.nn.functional.pad(out[:, _NUM_QUERIES:], (0, 0, 0, max(num_known) - count))
            for out, count in zip(per_frame, num_known)
        ],
        dim=1,
    )
    queries = torch.stack([out[:, :_NUM_QUERIES] for out in per_frame], dim=1)
    # The head makes no denoising queries without boxes.
    denoising = None
    if sum(num_known) > 0:
        denoising = StreamPETRDenoisingOutputs(
            cls_scores=known_outputs[..., :_NUM_CLASSES],
            bbox_preds=known_outputs[..., _NUM_CLASSES:],
            known_labels=torch.cat([labels.repeat(_SCALAR) for *_, labels in frames]),
            known_boxes=torch.cat([boxes.repeat(_SCALAR, 1) for _, _, boxes, _ in frames]),
            batch_indices=torch.cat(
                [
                    torch.full((count,), index, dtype=torch.long)
                    for index, count in enumerate(num_known)
                ]
            ),
            slot_indices=torch.cat([torch.arange(count) for count in num_known]),
        )
    head_outputs = StreamPETRHeadOutputs(
        all_cls_scores=queries[..., :_NUM_CLASSES],
        all_bbox_preds=queries[..., _NUM_CLASSES:],
        denoising=denoising,
    )
    return head.loss(
        head_outputs,
        [boxes for _, _, boxes, _ in frames],
        [labels for *_, labels in frames],
        padding_mask=torch.zeros(len(frames), dtype=torch.bool),
    )["loss"]


def _ddp_gradient_worker(
    rank: int, world_size: int, init_file: str, box_counts: tuple[int, ...]
) -> None:
    head = _build_head().train()
    frames = [_frame(count, seed=70 + 10 * index) for index, count in enumerate(box_counts)]
    torch.manual_seed(0)
    layer = torch.nn.Linear(_FEATURES, _NUM_CLASSES + 10)
    whole_batch_layer = copy.deepcopy(layer)
    # The reference runs before the process group exists: the whole batch on one process, its
    # counts floored as on one device.
    _layer_loss(head, whole_batch_layer, frames).backward()

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        # The wrapper must outlive the backward pass, which its hooks average over the ranks.
        ddp_layer = DistributedDataParallel(layer)
        _layer_loss(head, ddp_layer, [frames[rank]]).backward()

        for sharded, whole in zip(layer.parameters(), whole_batch_layer.parameters()):
            torch.testing.assert_close(sharded.grad, whole.grad, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


def test_reduce_mean_count_is_identity_without_process_group() -> None:
    """
    Input: a float and an integer count, without a process group.
    Expected: on one device the local count is the global count, so the float count comes back
    unchanged and the integer one is cast to float, as the distributed path casts it.
    Check: the float count is the same tensor, and the integer one comes back as float32.
    """
    assert reduce_mean_count(torch.tensor(3.0)).item() == pytest.approx(3.0)
    assert reduce_mean_count(torch.tensor(3)).dtype == torch.float32
    assert reduce_mean_count(torch.tensor(0.0)).item() == pytest.approx(1.0)


def test_reduce_mean_count_averages_counts_across_ranks() -> None:
    """
    Input: two CPU gloo ranks holding the counts 2 and 8.
    Expected: every rank gets the mean count, 5, and its own count is not modified in place.
    Check: the reduced count and the local count on every rank.
    """
    _spawn(_reduce_mean_count_worker)


def test_sharded_loss_matches_single_batch_equivalent() -> None:
    """
    Input: two CPU gloo ranks, rank 0 holding a frame with 2 boxes and rank 1 a frame with 8,
    and the same two frames as one batch on every rank.
    Expected: the losses are normalized by the object count averaged across the ranks, so the
    DDP average of the sharded losses is the loss of the whole batch on one device; rank-local
    counts would give every rank an equal vote whatever its object count.
    Check: the classification and regression losses averaged over the ranks against those of the
    combined batch, within a relative tolerance of 1e-5.
    """
    _spawn(_sharding_invariance_worker)


def test_dn_collective_survives_rank_without_ground_truth() -> None:
    """
    Input: two CPU gloo ranks, rank 0 with four ground-truth boxes and their denoising queries,
    rank 1 without ground truth and without denoising queries.
    Expected: a rank without ground truth still reaches the collective that counts the denoising
    queries, so a step mixing both does not deadlock, and it returns zero denoising losses under
    the same keys.
    Check: both ranks finish within the timeout with both denoising losses, which are zero on
    rank 1.
    """
    _spawn(_dn_mixed_gt_worker)


@pytest.mark.parametrize("box_counts", [(10, 30), (1, 0), (0, 0)], ids=["10-30", "1-0", "0-0"])
def test_ddp_gradients_match_the_whole_batch_on_one_process(box_counts: tuple[int, int]) -> None:
    """
    Input: two CPU gloo ranks, each holding a frame with the given number of boxes, every logit
    and box of the matching and the denoising queries made by one linear layer wrapped in DDP;
    and the same two frames as one batch through a copy of the layer on one process.
    Expected: DDP averages the gradients of the ranks, so normalizing by the counts averaged
    over the ranks gives the gradients of the whole batch on one process, for the matching and
    the denoising losses alike; rank-local counts would weight the two ranks equally. With one
    positive in the step, or none, the floor of the count applies to the whole batch: a floor
    of the mean would halve the gradient of the single positive, or of the negatives.
    Check: the DDP-averaged gradient of every parameter against the whole-batch gradient.
    """
    _spawn(_ddp_gradient_worker, box_counts)
