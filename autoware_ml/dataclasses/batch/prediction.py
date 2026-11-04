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

"""Samples of the prediction task.

The agent tracks and the vector map of a scenario are dataset agnostic: a reader of every
source format fills the same tensors, so the transforms and the models downstream never see
the format of the files.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Bool, Float32, Int64
from torch import Tensor


class AgentTracksSample(NamedTuple):
    """
    Named tuple to represent the agent tracks of one scenario over its time window, in the
    coordinate frame of the source (the city frame for Argoverse 2).

    Every tensor is indexed by agent first and by step second. A step an agent was not observed
    at is marked invalid and carries zeros.

    The box size of every agent is optional: a source that annotates 3D boxes (T4dataset) fills
    it, a source that records only trajectories (Argoverse 2) leaves it None. It is one size
    per agent, not per step, since the size of an annotated instance does not change over a
    scenario.
    """

    # (A, T, 3), position of every agent at every step
    positions: Float32[Tensor, "num_agents num_steps 3"]
    # (A, T), heading of every agent at every step in radians
    headings: Float32[Tensor, "num_agents num_steps"]
    # (A, T), planar velocity of every agent at every step in meters per second
    velocities: Float32[Tensor, "num_agents num_steps 2"]
    # (A, T), whether the agent was observed at the step
    valid: Bool[Tensor, "num_agents num_steps"]
    # (A, ), class index of every agent at the level of the taxonomy, the ignore index for an
    # agent the level drops, which stays in the tensors as context
    agent_types: Int64[Tensor, " num_agents"]
    # (A, ), role of every agent in the scenario, an AgentRole value
    agent_roles: Int64[Tensor, " num_agents"]
    # Source identifier of every agent, the focal agent first when the scenario has one
    track_ids: Sequence[str]
    # Index of the current step: steps up to it are observed, steps after it are the future
    current_step_index: int
    # Period between two steps in seconds
    step_seconds: float
    # Class index of an agent the taxonomy does not name
    ignore_index: int
    # (A, 3), length, width and height of every agent in meters, or None when the source
    # records no box size
    sizes: Float32[Tensor, "num_agents 3"] | None = None

    @property
    def num_agents(self) -> int:
        """Number of agents in the scenario."""
        return self.positions.shape[0]

    @property
    def num_steps(self) -> int:
        """Number of steps in the window."""
        return self.positions.shape[1]


class MapPolylinesSample(NamedTuple):
    """
    Named tuple to represent the vector map around one scenario as polylines of a fixed number
    of points, in the coordinate frame of the agent tracks.
    """

    # (P, L, 3), points of every polyline
    points: Float32[Tensor, "num_polylines num_points 3"]
    # (P, L), whether the point of the polyline is set
    valid: Bool[Tensor, "num_polylines num_points"]
    # (P, ), kind of every polyline, a PolylineType value
    polyline_types: Int64[Tensor, " num_polylines"]
    # (P, ), whether the polyline belongs to an intersection
    is_intersection: Bool[Tensor, " num_polylines"]
    # Source identifier of the map element every polyline comes from
    element_ids: Sequence[str]

    @property
    def num_polylines(self) -> int:
        """Number of polylines in the map."""
        return self.points.shape[0]

    @property
    def num_points_per_polyline(self) -> int:
        """Number of points of every polyline."""
        return self.points.shape[1]
