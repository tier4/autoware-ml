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

"""Tests for the SparseDrive agent motion head."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.dataclasses.batch.motion import AgentTracksBatch, AgentTracksSample
from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.models.sparsedrive.sparse4d_head import (
    ANCHOR_DIM,
    InstanceBank,
    SparseBox3DEncoder,
)
from autoware_ml.models.detection3d.task_modules.streaming import pos2posemb2d
from autoware_ml.models.sparsedrive.motion_head import MotionHead, agent_future_targets
from autoware_ml.models.sparsedrive.tests.motion_fixtures import build_motion_head

DIMS, QUERIES, CLASSES = 32, 12, 3


def random_anchor(batch_size: int) -> torch.Tensor:
    """Random anchors with a unit heading."""
    anchor = torch.randn(batch_size, QUERIES, 10)
    anchor[..., 6:8] = torch.nn.functional.normalize(anchor[..., 6:8], dim=-1)
    return anchor


def run(
    head: MotionHead,
    batch_size: int = 2,
    feature: torch.Tensor | None = None,
    encoder: SparseBox3DEncoder | None = None,
) -> MotionHeadOutputs:
    """One frame of random detections."""
    return head(
        torch.randn(batch_size, QUERIES, DIMS) if feature is None else feature,
        random_anchor(batch_size),
        torch.randn(batch_size, QUERIES, CLASSES),
        SparseBox3DEncoder(DIMS) if encoder is None else encoder,
        torch.zeros(batch_size, dtype=torch.bool),
        None,
        torch.eye(4).repeat(batch_size, 1, 1),
    )


class TestMotionHead(unittest.TestCase):
    def test_the_loss_is_finite_and_reaches_every_parameter(self) -> None:
        """
        Input: two frames, queries 0 and 1 matched to boxes with futures.
        Expected: finite mode and trajectory terms, two motion targets, a gradient for every
          parameter.
        """
        head = build_motion_head(CLASSES)
        outputs = run(head)
        gt = AgentFutureTargets(
            agent_future_positions=torch.randn(2, 3, 4, 2),
            agent_future_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )
        assigned = torch.full((2, QUERIES), -1)
        assigned[0, 0], assigned[1, 1] = 11, 12
        ids = torch.tensor([[11, -1, -1], [5, 12, -1]])
        losses = head.loss(outputs, assigned, torch.zeros(2, 3, 10), ids, gt)
        self.assertEqual(
            set(losses), {"loss_motion_cls", "loss_motion_reg", "num_motion_targets", "loss"}
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertEqual(losses["num_motion_targets"].item(), 2)
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters() if p.requires_grad))

    def test_the_trajectory_loss_sums_the_steps_and_averages_the_agents(self) -> None:
        """
        Input: two frames, one agent each at (3, 4) and (-2, 1), the first with 2 of 4 future
          steps known.
        Expected: the L1 error of the best mode against the future relative to the box center,
          summed over the known steps and axes of every agent, averaged over the 2 agents, times
          the weight.
        """
        head = build_motion_head(CLASSES)
        outputs = run(head)
        future = torch.randn(2, 1, 4, 2)
        valid = torch.ones(2, 1, 4, dtype=torch.bool)
        valid[0, 0, 2:] = False
        gt = AgentFutureTargets(agent_future_positions=future, agent_future_valid=valid)
        assigned = torch.full((2, QUERIES), -1)
        assigned[0, 3], assigned[1, 5] = 7, 8
        ids = torch.tensor([[7], [8]])
        boxes = torch.zeros(2, 1, 10)
        boxes[:, 0, :2] = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])

        losses = head.loss(outputs, assigned, boxes, ids, gt)

        expected = 0.0
        for frame, query in ((0, 3), (1, 5)):
            trajectories = outputs.agent_trajectories[frame, query].detach()
            known = valid[frame, 0]
            displacement = future[frame, 0] - boxes[frame, 0, :2]
            error = (trajectories - displacement).norm(dim=-1)[:, known].mean(-1)
            best = trajectories[error.argmin(), known]
            expected += (best - displacement[known]).abs().sum().item()
        torch.testing.assert_close(
            losses["loss_motion_reg"].item(), head.loss_motion_reg_weight * expected / 2
        )

    def test_the_mode_target_is_the_closest_mode(self) -> None:
        """
        Input: one frame with query 2 matched to an agent whose future one mode follows exactly.
        Expected: the mode classification targets that mode on the row of query 2 and weighs no
          other row.
        """
        head = build_motion_head(CLASSES)
        outputs = run(head, batch_size=1)
        trajectories = outputs.agent_trajectories.detach()
        future = trajectories[:, 2, 4][:, None].clone()
        gt = AgentFutureTargets(
            agent_future_positions=future,
            agent_future_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        )
        assigned = torch.full((1, QUERIES), -1)
        assigned[0, 2] = 9

        with mock.patch.object(head.loss_cls, "forward", wraps=head.loss_cls.forward) as loss_cls:
            head.loss(outputs, assigned, torch.zeros(1, 1, 10), torch.tensor([[9]]), gt)

        _, mode_targets = loss_cls.call_args.args
        weights = loss_cls.call_args.kwargs["weights"]
        self.assertEqual(mode_targets[2].argmax().item(), 4)
        self.assertEqual(mode_targets[2].sum().item(), 1.0)
        self.assertEqual(weights.sum(dim=1).nonzero().flatten().tolist(), [2])

    def test_a_query_on_a_box_without_a_known_future_is_no_target(self) -> None:
        """
        Input: one frame with queries 0 and 1 matched to two agents, the first without a known
          future step, then the same frame with query 0 unmatched.
        Expected: one target in both, the same trajectory loss.
        """
        head = build_motion_head(CLASSES)
        outputs = run(head, batch_size=1)
        valid = torch.ones(1, 2, 4, dtype=torch.bool)
        valid[0, 0] = False
        gt = AgentFutureTargets(
            agent_future_positions=torch.randn(1, 2, 4, 2), agent_future_valid=valid
        )
        ids = torch.tensor([[5, 6]])
        both = torch.full((1, QUERIES), -1)
        both[0, :2] = torch.tensor([5, 6])
        second = torch.full((1, QUERIES), -1)
        second[0, 1] = 6

        with_unknown = head.loss(outputs, both, torch.zeros(1, 2, 10), ids, gt)
        without = head.loss(outputs, second, torch.zeros(1, 2, 10), ids, gt)

        self.assertEqual(with_unknown["num_motion_targets"].item(), 1.0)
        torch.testing.assert_close(with_unknown["loss_motion_reg"], without["loss_motion_reg"])

    def test_a_loss_over_no_frame_reports_every_term_as_zero(self) -> None:
        """
        Input: the outputs of two frames, neither of them supervised.
        Expected: the terms of a supervised loss, zero and still in the graph.
        """
        head = build_motion_head(CLASSES)
        none = torch.zeros(2, dtype=torch.bool)
        gt = AgentFutureTargets(
            agent_future_positions=torch.randn(2, 3, 4, 2),
            agent_future_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )

        losses = head.loss(
            run(head).select_samples(none),
            torch.full((0, QUERIES), -1),
            torch.zeros(0, 3, 10),
            torch.zeros(0, 3, dtype=torch.long),
            gt.select_samples(none),
        )

        self.assertEqual(
            list(losses), ["loss_motion_cls", "loss_motion_reg", "num_motion_targets", "loss"]
        )
        self.assertTrue(all(value.item() == 0.0 for value in losses.values()))
        losses["loss"].backward()

    def test_a_detached_head_trains_without_touching_the_detection(self) -> None:
        """
        Input: a head built with detach_detection run on detection features and an anchor encoder
          that track gradients, then its outputs backpropagated.
        Expected: the head parameters get gradients, the detection features and the anchor encoder
          none.
        """
        head = build_motion_head(CLASSES, detach_detection=True)
        encoder = SparseBox3DEncoder(DIMS)
        feature = torch.randn(1, QUERIES, DIMS, requires_grad=True)
        run(
            head, batch_size=1, feature=feature, encoder=encoder
        ).agent_trajectories.sum().backward()

        self.assertIsNone(feature.grad)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertIsNotNone(head.motion_reg[-1].weight.grad)

    def test_a_query_carried_into_the_next_frame_reads_its_history(self) -> None:
        """
        Input: two frames of one lane, the second carrying its first temporal slot from query 0, run
          once after a first frame with other features.
        Expected: the trajectories of the carried query depend on the first frame, those of the
          fresh queries do not.
        """
        head = build_motion_head(CLASSES).eval()
        anchor = random_anchor(1)
        classes = torch.randn(1, QUERIES, CLASSES)
        encoder = SparseBox3DEncoder(DIMS)
        second = torch.randn(1, QUERIES, DIMS)
        trajectories = []
        for seed in range(2):
            head.reset()
            first = torch.randn(1, QUERIES, DIMS, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                for feature, continues in ((first, False), (second, True)):
                    outputs = head(
                        feature,
                        anchor,
                        classes,
                        encoder,
                        torch.tensor([continues]),
                        torch.zeros(1, 1, dtype=torch.long),
                        torch.eye(4)[None],
                    )
            trajectories.append(outputs.agent_trajectories)

        self.assertFalse(torch.allclose(trajectories[0][:, 0], trajectories[1][:, 0]))
        torch.testing.assert_close(trajectories[0][:, 1:], trajectories[1][:, 1:])

    def test_a_new_batch_size_drops_the_carried_slots(self) -> None:
        """
        Input: frames of 8, 8 and 1 lanes in the order of the motion model, the detection bank
          first, then the motion head with the slot index the bank carried into the frame.
        Expected: every frame runs, the one lane frame reads no slot of the eight lane batch, so
          it predicts what a head without history predicts.
        """
        anchors = np.zeros((QUERIES, ANCHOR_DIM), dtype=np.float32)
        anchors[:, 7] = 1.0
        bank = fill_anchors(
            InstanceBank(
                num_anchor=QUERIES,
                embed_dims=DIMS,
                num_temp_instances=4,
                anchor_recipe=FixedAnchorRecipe({"anchors": anchors}),
            )
        )
        head = build_motion_head(CLASSES).eval()
        fresh = copy.deepcopy(head)
        encoder = SparseBox3DEncoder(DIMS)
        for step, batch_size in enumerate((8, 8, 1)):
            timestamps = torch.full((batch_size,), 0.1 * step, dtype=torch.float64)
            poses = torch.eye(4).repeat(batch_size, 1, 1)
            bank.get(batch_size, timestamps, poses, [f"scene{lane}" for lane in range(batch_size)])
            feature = torch.randn(batch_size, QUERIES, DIMS)
            anchor = bank.anchor.detach()[None].expand(batch_size, -1, -1).clone()
            logits = torch.randn(batch_size, QUERIES, CLASSES)
            bank.cache(feature, anchor, logits, poses)
            carried_index = bank.carried_query_index
            with torch.no_grad():
                outputs = head(
                    feature, anchor, logits, encoder, bank.lane_continues, carried_index, poses
                )

            self.assertEqual(outputs.agent_trajectories.shape[0], batch_size)
        with torch.no_grad():
            reference = fresh(feature, anchor, logits, encoder, torch.tensor([False]), None, poses)
        torch.testing.assert_close(outputs.agent_trajectories, reference.agent_trajectories)

    def test_the_mode_anchors_turn_with_the_heading(self) -> None:
        """
        Input: every mode anchor ending at (1, 0), query 0 heading along y and query 1 with a
          zero heading.
        Expected: the mode queries embed the endpoint (0, 1) for query 0 and the unturned (1, 0)
          for query 1, the frame of the anchors turns with the heading of the agent.
        """
        head = build_motion_head(CLASSES).eval()
        head.motion_anchors[:, :, -1] = torch.tensor([1.0, 0.0])
        anchor = random_anchor(1)
        anchor[0, 0, 6:8] = torch.tensor([1.0, 0.0])
        anchor[0, 1, 6:8] = 0.0
        embedded = []
        hook = head.motion_mode_fc.register_forward_hook(
            lambda module, inputs, output: embedded.append(inputs[0])
        )
        self.addCleanup(hook.remove)

        with torch.no_grad():
            head._predict_agents(
                torch.randn(1, QUERIES, DIMS), anchor, torch.randn(1, QUERIES, CLASSES)
            )

        for query, endpoint in ((0, [0.0, 1.0]), (1, [1.0, 0.0])):
            expected = pos2posemb2d(torch.tensor([endpoint]), DIMS // 2)
            torch.testing.assert_close(
                embedded[0][0, query], expected.expand(6, -1), atol=1e-5, rtol=0.0
            )

    def test_only_the_angle_of_the_heading_turns_the_mode_anchors(self) -> None:
        """
        Input: the same queries with unit headings and with the headings shrunk to 60 %.
        Expected: the regressed heading is not unit length, only its angle turns the anchors.
        Check: equal mode logits and trajectories.
        """
        head = build_motion_head(CLASSES).eval()
        context = torch.randn(2, QUERIES, DIMS)
        anchor = random_anchor(2)
        shrunk = anchor.clone()
        shrunk[..., 6:8] *= 0.6
        cls_logits = torch.randn(2, QUERIES, CLASSES)

        with torch.no_grad():
            unit_logits, unit_trajectories = head._predict_agents(context, anchor, cls_logits)
            shrunk_logits, shrunk_trajectories = head._predict_agents(context, shrunk, cls_logits)

        torch.testing.assert_close(shrunk_logits, unit_logits)
        torch.testing.assert_close(shrunk_trajectories, unit_trajectories)

    def test_the_model_refuses_futures_of_other_steps_than_its_head(self) -> None:
        """
        Input: a batch with tracks of 12 future steps for a head that predicts 6, and a batch
          without tracks.
        Expected: a ValueError for both, a gather over the steps would truncate the futures.
        """
        motion_head = SimpleNamespace(agent_steps=6)
        agents = AgentTracksBatch.collate(
            [
                AgentTracksSample(
                    positions=torch.zeros(2, 13, 3),
                    headings=torch.zeros(2, 13),
                    velocities=torch.zeros(2, 13, 2),
                    valid=torch.ones(2, 13, dtype=torch.bool),
                    agent_types=torch.zeros(2, dtype=torch.int64),
                    agent_roles=torch.zeros(2, dtype=torch.int64),
                    track_ids=["a", "b"],
                    current_step_index=0,
                    step_seconds=0.5,
                    ignore_index=-1,
                )
            ]
        )

        def batch(motion_gt: SimpleNamespace | None) -> SimpleNamespace:
            return SimpleNamespace(
                multi_task_gt_batch=SimpleNamespace(
                    motion_gt_batch=motion_gt,
                    detection3d_gt_batch=SimpleNamespace(
                        gt_bboxes_instance_ids=torch.tensor([[-1, -1]])
                    ),
                )
            )

        with self.assertRaisesRegex(ValueError, "futures of 12 steps, the motion head predicts 6"):
            agent_future_targets(motion_head, batch(SimpleNamespace(agents=agents)))
        with self.assertRaisesRegex(ValueError, "agent tracks of every sample"):
            agent_future_targets(motion_head, batch(None))
