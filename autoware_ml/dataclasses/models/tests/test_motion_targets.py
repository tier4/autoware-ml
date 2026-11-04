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

"""Tests for the agent futures aligned with the 3D box slots."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.motion import AgentTracksBatch, AgentTracksSample
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids


def tracks(track_ids: list[str], valid: list[list[bool]]) -> AgentTracksSample:
    """Tracks of a current step and two futures, agent ``a`` at x = 10 a + step."""
    num_agents = len(track_ids)
    steps = torch.arange(3, dtype=torch.float32)
    positions = torch.zeros(num_agents, 3, 3)
    positions[..., 0] = 10.0 * torch.arange(num_agents, dtype=torch.float32)[:, None] + steps
    return AgentTracksSample(
        positions=positions,
        headings=torch.zeros(num_agents, 3),
        velocities=torch.zeros(num_agents, 3, 2),
        valid=torch.tensor(valid, dtype=torch.bool).reshape(num_agents, 3),
        agent_types=torch.zeros(num_agents, dtype=torch.int64),
        agent_roles=torch.zeros(num_agents, dtype=torch.int64),
        track_ids=track_ids,
        current_step_index=0,
        step_seconds=0.5,
        ignore_index=-1,
    )


def slot_ids(*tokens: str | None) -> list[int]:
    """Instance ids of box slots, -1 for a padding slot."""
    return [-1 if token is None else int(hash_instance_ids([token])[0]) for token in tokens]


class TestAgentFutureTargets(unittest.TestCase):
    def test_futures_follow_the_box_slots_by_instance_id(self) -> None:
        """
        Input: frame 0 with agents a, b and boxes b, a, padding, and frame 1 with agent c, its
          second future unobserved, and boxes c, an unknown box.
        Expected: every box slot holds the future steps of its agent, the unknown and the
          padding slots and the unobserved step are invalid and zero.
        """
        agents = AgentTracksBatch.collate(
            [
                tracks(["a", "b"], [[True] * 3, [True] * 3]),
                tracks(["c"], [[True, True, False]]),
            ]
        )
        ids = torch.tensor([slot_ids("b", "a", None), slot_ids("c", "x", None)])

        targets = AgentFutureTargets.from_agent_tracks(agents, ids)

        torch.testing.assert_close(
            targets.agent_future_positions[0, :2, :, 0], torch.tensor([[11.0, 12.0], [1.0, 2.0]])
        )
        self.assertEqual(
            targets.agent_future_valid.tolist(),
            [
                [[True, True], [True, True], [False, False]],
                [[True, False], [False] * 2, [False] * 2],
            ],
        )
        self.assertEqual(float(targets.agent_future_positions[1].abs().sum()), 1.0)

    def test_a_batch_without_agents_has_no_valid_future(self) -> None:
        """
        Input: one frame without agents and two box slots.
        Expected: zero futures of the two slots, none valid.
        """
        agents = AgentTracksBatch.collate([tracks([], [])])

        targets = AgentFutureTargets.from_agent_tracks(agents, torch.tensor([slot_ids("a", None)]))

        self.assertEqual(tuple(targets.agent_future_positions.shape), (1, 2, 2, 2))
        self.assertFalse(bool(targets.agent_future_valid.any()))

    def test_tracks_and_boxes_of_other_batch_sizes_are_rejected(self) -> None:
        """
        Input: the tracks of one frame and the box slots of two.
        Expected: ValueError, the frames cannot be paired.
        """
        agents = AgentTracksBatch.collate([tracks(["a"], [[True] * 3])])

        with self.assertRaisesRegex(ValueError, "tracks of 1 samples and the boxes of 2"):
            AgentFutureTargets.from_agent_tracks(agents, torch.tensor([[-1], [-1]]))

    def test_select_samples_keeps_the_masked_frames(self) -> None:
        """
        Input: the targets of two frames and a mask keeping the second.
        Expected: the targets of the second frame only.
        """
        agents = AgentTracksBatch.collate(
            [tracks(["a"], [[True] * 3]), tracks(["b"], [[True, False, True]])]
        )
        targets = AgentFutureTargets.from_agent_tracks(
            agents, torch.tensor([slot_ids("a"), slot_ids("b")])
        )

        kept = targets.select_samples(torch.tensor([False, True]))

        self.assertEqual(kept.agent_future_valid.tolist(), [[[False, True]]])
