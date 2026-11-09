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

"""Tests for the prediction samples."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.prediction import AgentTracksSample


def _tracks(num_agents: int = 2, num_steps: int = 5, **kwargs) -> AgentTracksSample:
    """Build a zero filled agent tracks sample with the given fields overridden."""
    return AgentTracksSample(
        positions=torch.zeros(num_agents, num_steps, 3),
        headings=torch.zeros(num_agents, num_steps),
        velocities=torch.zeros(num_agents, num_steps, 2),
        valid=torch.ones(num_agents, num_steps, dtype=torch.bool),
        agent_types=torch.zeros(num_agents, dtype=torch.int64),
        agent_roles=torch.zeros(num_agents, dtype=torch.int64),
        track_ids=[str(i) for i in range(num_agents)],
        current_step_index=2,
        step_seconds=0.1,
        ignore_index=-1,
        **kwargs,
    )


class TestAgentTracksSample(unittest.TestCase):
    """The agent tracks sample counts its agents and steps and carries optional sizes."""

    def test_agent_tracks_sizes_default_to_none(self) -> None:
        """
        Input: a sample of two agents over five steps built without sizes.
        Expected: the counts follow the tensor shapes and the sizes default to None, since
        not every dataset records a box size.
        Check: num_agents, num_steps and sizes.
        """
        tracks = _tracks()

        self.assertEqual(tracks.num_agents, 2)
        self.assertEqual(tracks.num_steps, 5)
        self.assertIsNone(tracks.sizes)

    def test_agent_tracks_carry_one_size_per_agent(self) -> None:
        """
        Input: the sample built with a float32 size per agent.
        Expected: the sizes are stored as given, one row per agent, and can be dropped again
        through _replace.
        Check: the presence, shape and dtype of sizes and the sizes of the replaced sample.
        """
        sizes = torch.tensor([[4.5, 1.8, 1.5], [0.6, 0.6, 1.7]])

        tracks = _tracks(sizes=sizes)

        self.assertIsNotNone(tracks.sizes)
        self.assertEqual(tracks.sizes.shape, (tracks.num_agents, 3))
        self.assertEqual(tracks.sizes.dtype, torch.float32)
        self.assertIsNone(tracks._replace(sizes=None).sizes)
