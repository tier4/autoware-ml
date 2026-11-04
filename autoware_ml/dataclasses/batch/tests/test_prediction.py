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

from __future__ import annotations

import torch

from autoware_ml.dataclasses.batch.prediction import AgentTracksSample


def _tracks(num_agents: int = 2, num_steps: int = 5, **kwargs) -> AgentTracksSample:
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


def test_agent_tracks_sizes_default_to_none() -> None:
    tracks = _tracks()

    assert tracks.num_agents == 2
    assert tracks.num_steps == 5
    assert tracks.sizes is None


def test_agent_tracks_carry_one_size_per_agent() -> None:
    sizes = torch.tensor([[4.5, 1.8, 1.5], [0.6, 0.6, 1.7]])

    tracks = _tracks(sizes=sizes)

    assert tracks.sizes is not None
    assert tracks.sizes.shape == (tracks.num_agents, 3)
    assert tracks.sizes.dtype == torch.float32
    assert tracks._replace(sizes=None).sizes is None
