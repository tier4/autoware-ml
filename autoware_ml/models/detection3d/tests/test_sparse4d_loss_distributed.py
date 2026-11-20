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

"""Multi-rank tests of the Sparse4D loss normalization.

They spawn gloo process groups on CPU so the collectives of ``reduce_mean_count`` run. Every rank
runs the same head on the same two frame batch and computes the loss of the frames a data parallel
run hands it. The rank average of these losses and of their gradients equals the loss of the
supervised frames on one process, and a rank without a supervised frame still joins the
collectives.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence

import torch
import torch.distributed as dist
from jaxtyping import Float

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Sparse4DHeadOutputs
from autoware_ml.models.detection3d.heads.sparse4d import Sparse4DHead
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    forward_frame,
    gt_batch,
    head_loss,
    make_anchor_recipe,
    random_stage,
)
from autoware_ml.utils.tests.distributed_fixtures import GLOO_AVAILABLE, spawn_gloo_workers

_POINT_CLOUD_RANGE = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]
# The stage grid of the range at a stride of 2
_STAGE_DIMS = (51, 51, 4)
_BOXES = (
    torch.tensor(
        [
            [5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0],
            [-10.0, 8.0, 0.2, 1.0, 0.8, 1.7, 1.2, 0.0, 0.0],
        ]
    ),
    torch.tensor([[12.0, 3.0, 0.5, 4.5, 1.9, 1.7, -0.3, 0.0, 0.0]]),
)
_LABELS = (torch.tensor([0, 1]), torch.tensor([2]))


def _forward_batch() -> tuple[Sparse4DHead, Sparse4DHeadOutputs, Detection3DGTBatch]:
    """The same denoising head and two frame batch on every rank, run in training mode."""
    torch.manual_seed(0)
    head = build_sparse4d_head(
        make_anchor_recipe(16, -40.0, 40.0),
        num_classes=3,
        num_anchor=16,
        point_cloud_range=_POINT_CLOUD_RANGE,
        feature_strides=(2,),
        num_temp_instances=8,
        dn_query_budget=12,
    ).train()
    gt = gt_batch(list(_BOXES), list(_LABELS))
    stages = [random_stage(_STAGE_DIMS, 128, batch_size=2)]
    return head, forward_frame(head, stages, gt_detections=gt), gt


def _lane_loss(
    head: Sparse4DHead,
    outputs: Sparse4DHeadOutputs,
    gt: Detection3DGTBatch,
    lanes: Sequence[int],
) -> Float[torch.Tensor, ""]:
    """Total loss of some lanes of the batch, the outputs of the other lanes left out."""
    mask = torch.zeros(outputs.classification.shape[0], dtype=torch.bool)
    mask[list(lanes)] = True
    return head_loss(head, outputs.select_samples(mask), gt, lanes)["loss"]


def _gradients(
    head: Sparse4DHead, loss: Float[torch.Tensor, ""]
) -> list[Float[torch.Tensor, "..."]]:
    """Gradient of the loss for every parameter of the head, zero for an unreached one."""
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]


def _rank_average(value: Float[torch.Tensor, "..."], world_size: int) -> Float[torch.Tensor, "..."]:
    """Average of a tensor over the ranks, as the gradient averaging of data parallel does."""
    total = value.detach().clone()
    dist.all_reduce(total)
    return total / world_size


def _sharding_worker(rank: int, world_size: int, rank_lanes: tuple[tuple[int, ...], ...]) -> None:
    head, outputs, gt = _forward_batch()
    sharded = _lane_loss(head, outputs, gt, rank_lanes[rank])
    sharded_gradients = _gradients(head, sharded)
    # Every rank computes the same reference, its counts reduce to themselves
    supervised = sorted(lane for lanes in rank_lanes for lane in lanes)
    whole = _lane_loss(head, outputs, gt, supervised)
    whole_gradients = _gradients(head, whole)

    case = unittest.TestCase()
    case.assertTrue(bool(torch.isfinite(sharded)))
    torch.testing.assert_close(_rank_average(sharded, world_size), whole.detach())
    for sharded_gradient, whole_gradient in zip(sharded_gradients, whole_gradients, strict=True):
        torch.testing.assert_close(
            _rank_average(sharded_gradient, world_size), whole_gradient, rtol=1e-4, atol=1e-6
        )


@unittest.skipUnless(GLOO_AVAILABLE, "gloo backend unavailable")
class TestSparse4DLossDistributed(unittest.TestCase):
    def test_the_rank_average_of_one_frame_per_rank_is_the_loss_of_the_batch(self) -> None:
        """
        Input: two ranks, each supervising one frame of a two frame batch, the frames holding two
          boxes and one box.
        Expected: the rank average of the losses and of their gradients equals the loss of both
          frames on one process. Rank local counts would weigh both frames alike whatever their
          box counts.
        """
        spawn_gloo_workers(_sharding_worker, ((0,), (1,)))

    def test_a_rank_without_a_supervised_frame_joins_the_collectives(self) -> None:
        """
        Input: two ranks, the first supervising the first frame, the second none.
        Expected: both ranks complete the step, the second with a finite loss, and the rank
          average equals the loss of the first frame on one process.
        """
        spawn_gloo_workers(_sharding_worker, ((0,), ()))
