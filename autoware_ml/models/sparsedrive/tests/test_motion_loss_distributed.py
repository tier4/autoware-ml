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

"""Multi-rank tests of the motion loss normalization.

They spawn gloo process groups on CPU so the collectives of ``reduce_mean_count`` run. Every rank
holds the same outputs of a two frame batch and computes the loss of the frame a data parallel run
hands it. The rank average of these losses and of their gradients equals the loss of the batch on
one process, and a rank without a target still joins the collectives.
"""

from __future__ import annotations

import unittest

import torch
import torch.distributed as dist
from jaxtyping import Float, Int64

from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.models.sparsedrive.motion_head import MotionHead
from autoware_ml.models.sparsedrive.tests.motion_fixtures import build_motion_head
from autoware_ml.utils.tests.distributed_fixtures import GLOO_AVAILABLE, spawn_gloo_workers

_CLASSES, _QUERIES, _SLOTS = 3, 8, 3


def _frame_loss(
    head: MotionHead,
    outputs: MotionHeadOutputs,
    assigned: Int64[torch.Tensor, "batch_size num_queries"],
    targets: AgentFutureTargets,
    frames: list[int],
) -> Float[torch.Tensor, ""]:
    """Total motion loss of some frames of the batch."""
    mask = torch.zeros(assigned.shape[0], dtype=torch.bool)
    mask[frames] = True
    ids = torch.arange(1, _SLOTS + 1).expand(assigned.shape[0], -1)
    return head.loss(
        outputs.select_samples(mask),
        assigned[mask],
        torch.zeros(int(mask.sum()), _SLOTS, 10),
        ids[mask],
        targets.select_samples(mask),
    )["loss"]


def _sharding_worker(rank: int, world_size: int, matched: tuple[int, ...]) -> None:
    head = build_motion_head(_CLASSES)
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, _QUERIES, 6, generator=generator, requires_grad=True)
    trajectories = torch.randn(2, _QUERIES, 6, 4, 2, generator=generator, requires_grad=True)
    outputs = MotionHeadOutputs(agent_mode_logits=logits, agent_trajectories=trajectories)
    targets = AgentFutureTargets(
        agent_future_positions=torch.randn(2, _SLOTS, 4, 2, generator=generator),
        agent_future_valid=torch.ones(2, _SLOTS, 4, dtype=torch.bool),
    )
    # The first queries of every frame are matched to the box slots, as many as the frame names
    assigned = torch.full((2, _QUERIES), -1)
    for frame, count in enumerate(matched):
        assigned[frame, :count] = torch.arange(1, count + 1)

    sharded = _frame_loss(head, outputs, assigned, targets, [rank])
    sharded_gradients = torch.autograd.grad(sharded, (logits, trajectories))
    # Every rank computes the same reference, its counts reduce to themselves
    whole = _frame_loss(head, outputs, assigned, targets, [0, 1])
    whole_gradients = torch.autograd.grad(whole, (logits, trajectories))

    case = unittest.TestCase()
    case.assertTrue(bool(torch.isfinite(sharded)))
    for value, reference in zip(
        (sharded, *sharded_gradients), (whole, *whole_gradients), strict=True
    ):
        total = value.detach().clone()
        dist.all_reduce(total)
        torch.testing.assert_close(total / world_size, reference.detach(), rtol=1e-4, atol=1e-6)


@unittest.skipUnless(GLOO_AVAILABLE, "gloo backend unavailable")
class TestMotionLossDistributed(unittest.TestCase):
    def test_the_rank_average_of_one_frame_per_rank_is_the_loss_of_the_batch(self) -> None:
        """
        Input: two ranks, each supervising one frame of a two frame batch with three and one
          agent targets.
        Expected: the rank average of the losses and of their gradients equals the loss of both
          frames on one process.
        """
        spawn_gloo_workers(_sharding_worker, (3, 1))

    def test_a_rank_without_a_target_joins_the_collectives(self) -> None:
        """
        Input: two ranks, the frame of the second without a matched agent.
        Expected: both ranks complete the step, the second with a finite loss, and the rank
          average equals the loss of both frames on one process.
        """
        spawn_gloo_workers(_sharding_worker, (2, 0))
