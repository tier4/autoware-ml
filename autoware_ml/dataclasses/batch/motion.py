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

"""Samples of the motion task.

The agent tracks and the vector map of a scenario are dataset agnostic: a reader of every
source format fills the same tensors, so the transforms and the models downstream never see
the format of the files.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Bool, Float32, Int64
import torch
from torch import Tensor

from autoware_ml.types.dataset import DatasetFormat

# Role of a padded agent slot and type of a padded polyline slot, outside every enum value
PADDED_INDEX = -1


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


class PredictionSourceSample(NamedTuple):
    """
    Named tuple to represent the source of one scenario, its file paths resolved against the
    root of the dataset serving it: the analog of LiDARPointCloudSample for the prediction
    task. The loading transforms turn it into a PredictionGTSample.
    """

    # Dataset family of the scenario, picks the readers
    dataset_format: DatasetFormat
    # Path of the agent tracks file, None for a recording dataset
    tracks_path: str | None
    # Path of the vector map file, None for a recording dataset
    map_path: str | None
    # Key of the scenario inside a container file, None when the scenario has its own files
    scenario_key: str | None
    # Identifier of the focal agent, None when the scenario has none
    focal_track_id: str | None
    # Index of the first step of the window inside the source
    first_step_index: int
    # Index of the current step: steps up to it are observed, steps after it are the future
    current_step_index: int
    # Number of steps in the window
    num_steps: int
    # Period between two steps in seconds
    step_seconds: float
    # Number of agent tracks in the window
    num_tracks: int


class PredictionGTSample(NamedTuple):
    """Named tuple to represent the ground truth of the prediction task for one scenario."""

    # The agent tracks over the window of the scenario
    agents: AgentTracksSample
    # The vector map of the scenario, None when it was not loaded or the scenario has none
    map_polylines: MapPolylinesSample | None = None


def _pad_rows(tensor: Tensor, num_rows: int, fill: float | int | bool) -> Tensor:
    """
    Pad the first dimension of a tensor up to a row count.

    Args:
      tensor: Tensor to pad.
      num_rows: Row count to pad to, at least the rows of the tensor.
      fill: Value of the padded rows.

    Returns:
      Tensor: The padded tensor.
    """

    padding = num_rows - tensor.shape[0]
    if padding == 0:
        return tensor
    pad = torch.full((padding, *tensor.shape[1:]), fill, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


class PredictionGTBatch(NamedTuple):
    """
    Named tuple to represent the ground truth of the prediction task for a batch of scenarios.

    The agents are padded to the largest agent count of the batch and the polylines to the
    largest polyline count, with one mask each naming the real slots. A padded agent carries
    zeros, an invalid track, the ignore index as type and PADDED_INDEX as role; a padded
    polyline carries zeros, invalid points and PADDED_INDEX as type. Every scenario of a batch
    shares the window shape, so it is stored once.
    """

    # (B, A, T, 3), position of every agent at every step
    agent_positions: Float32[Tensor, "batch_size num_agents num_steps 3"]
    # (B, A, T), heading of every agent at every step in radians
    agent_headings: Float32[Tensor, "batch_size num_agents num_steps"]
    # (B, A, T, 2), planar velocity of every agent at every step
    agent_velocities: Float32[Tensor, "batch_size num_agents num_steps 2"]
    # (B, A, T), whether the agent was observed at the step
    agent_valid: Bool[Tensor, "batch_size num_agents num_steps"]
    # (B, A), class index of every agent, the ignore index for a dropped or padded agent
    agent_types: Int64[Tensor, "batch_size num_agents"]
    # (B, A), AgentRole value of every agent, PADDED_INDEX for a padded slot
    agent_roles: Int64[Tensor, "batch_size num_agents"]
    # (B, A), whether the agent slot holds an agent of the scenario
    agent_mask: Bool[Tensor, "batch_size num_agents"]
    # Source identifier of the agents of every scenario, the real slots only
    track_ids: Sequence[Sequence[str]]
    # Index of the current step, shared by every scenario of the batch
    current_step_index: int
    # Period between two steps in seconds, shared by every scenario of the batch
    step_seconds: float
    # Class index of an agent the taxonomy does not name
    ignore_index: int
    # (B, A, 3), size of every agent, None when the samples carry no sizes
    agent_sizes: Float32[Tensor, "batch_size num_agents 3"] | None = None
    # (B, P, L, 3), points of every polyline, None when the samples carry no map
    map_points: Float32[Tensor, "batch_size num_polylines num_points 3"] | None = None
    # (B, P, L), whether the point of the polyline is set
    map_valid: Bool[Tensor, "batch_size num_polylines num_points"] | None = None
    # (B, P), PolylineType value of every polyline, PADDED_INDEX for a padded slot
    map_polyline_types: Int64[Tensor, "batch_size num_polylines"] | None = None
    # (B, P), whether the polyline belongs to an intersection
    map_is_intersection: Bool[Tensor, "batch_size num_polylines"] | None = None
    # (B, P), whether the polyline slot holds a polyline of the scenario
    polyline_mask: Bool[Tensor, "batch_size num_polylines"] | None = None
    # Source identifier of the polylines of every scenario, the real slots only
    element_ids: Sequence[Sequence[str]] | None = None

    @property
    def batch_size(self) -> int:
        """Number of collated scenarios."""
        return int(self.agent_mask.shape[0])

    @property
    def has_map(self) -> bool:
        """Whether the batch carries the vector maps of its scenarios."""
        return self.map_points is not None

    @staticmethod
    def collate_gt_samples(
        prediction_gt_samples: Sequence[PredictionGTSample],
    ) -> PredictionGTBatch | None:
        """
        Collate a sequence of prediction GT samples into a single PredictionGTBatch.

        Args:
          prediction_gt_samples: Sequence of PredictionGTSample to be collated.

        Returns:
          PredictionGTBatch: Collated prediction GT batch, None for an empty sequence.

        Raises:
          ValueError: If the samples disagree on the window shape, the ignore index, the
            presence of sizes or of a map, or the points per polyline.
        """

        if len(prediction_gt_samples) == 0:
            return None

        agents = [sample.agents for sample in prediction_gt_samples]
        first = agents[0]
        for index, sample_agents in enumerate(agents):
            window = (
                sample_agents.num_steps,
                sample_agents.current_step_index,
                sample_agents.step_seconds,
            )
            if window != (first.num_steps, first.current_step_index, first.step_seconds):
                raise ValueError(
                    f"Sample {index} has a window of {window} (num_steps, current_step_index, "
                    f"step_seconds), sample 0 has {(first.num_steps, first.current_step_index, first.step_seconds)}; "
                    "every scenario of a batch shares the window shape."
                )
            if sample_agents.ignore_index != first.ignore_index:
                raise ValueError(
                    f"Sample {index} has the ignore index {sample_agents.ignore_index}, sample 0 "
                    f"has {first.ignore_index}; every scenario of a batch shares the taxonomy."
                )
        with_sizes = [sample_agents.sizes is not None for sample_agents in agents]
        if any(with_sizes) and not all(with_sizes):
            raise ValueError("Either every sample carries agent sizes or none does.")

        num_agents = max(sample_agents.num_agents for sample_agents in agents)
        batch = PredictionGTBatch(
            agent_positions=torch.stack([_pad_rows(a.positions, num_agents, 0.0) for a in agents]),
            agent_headings=torch.stack([_pad_rows(a.headings, num_agents, 0.0) for a in agents]),
            agent_velocities=torch.stack(
                [_pad_rows(a.velocities, num_agents, 0.0) for a in agents]
            ),
            agent_valid=torch.stack([_pad_rows(a.valid, num_agents, False) for a in agents]),
            agent_types=torch.stack(
                [_pad_rows(a.agent_types, num_agents, first.ignore_index) for a in agents]
            ),
            agent_roles=torch.stack(
                [_pad_rows(a.agent_roles, num_agents, PADDED_INDEX) for a in agents]
            ),
            agent_mask=torch.stack(
                [
                    _pad_rows(torch.ones(a.num_agents, dtype=torch.bool), num_agents, False)
                    for a in agents
                ]
            ),
            track_ids=[list(a.track_ids) for a in agents],
            current_step_index=first.current_step_index,
            step_seconds=first.step_seconds,
            ignore_index=first.ignore_index,
            agent_sizes=(
                torch.stack([_pad_rows(a.sizes, num_agents, 0.0) for a in agents])
                if all(with_sizes)
                else None
            ),
        )

        maps = [sample.map_polylines for sample in prediction_gt_samples]
        with_map = [polylines is not None for polylines in maps]
        if not any(with_map):
            return batch
        if not all(with_map):
            raise ValueError("Either every sample carries a vector map or none does.")
        num_points = maps[0].num_points_per_polyline
        for index, polylines in enumerate(maps):
            if polylines.num_points_per_polyline != num_points:
                raise ValueError(
                    f"Sample {index} has {polylines.num_points_per_polyline} points per "
                    f"polyline, sample 0 has {num_points}; every map of a batch shares it."
                )
        num_polylines = max(polylines.num_polylines for polylines in maps)
        return batch._replace(
            map_points=torch.stack([_pad_rows(m.points, num_polylines, 0.0) for m in maps]),
            map_valid=torch.stack([_pad_rows(m.valid, num_polylines, False) for m in maps]),
            map_polyline_types=torch.stack(
                [_pad_rows(m.polyline_types, num_polylines, PADDED_INDEX) for m in maps]
            ),
            map_is_intersection=torch.stack(
                [_pad_rows(m.is_intersection, num_polylines, False) for m in maps]
            ),
            polyline_mask=torch.stack(
                [
                    _pad_rows(torch.ones(m.num_polylines, dtype=torch.bool), num_polylines, False)
                    for m in maps
                ]
            ),
            element_ids=[list(m.element_ids) for m in maps],
        )

    def to_device(self, device: torch.device) -> PredictionGTBatch:
        """
        Move the PredictionGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          PredictionGTBatch: The batch moved to the specified device.
        """

        def move(tensor: Tensor | None) -> Tensor | None:
            return tensor.to(device) if tensor is not None else None

        return self._replace(
            agent_positions=self.agent_positions.to(device),
            agent_headings=self.agent_headings.to(device),
            agent_velocities=self.agent_velocities.to(device),
            agent_valid=self.agent_valid.to(device),
            agent_types=self.agent_types.to(device),
            agent_roles=self.agent_roles.to(device),
            agent_mask=self.agent_mask.to(device),
            agent_sizes=move(self.agent_sizes),
            map_points=move(self.map_points),
            map_valid=move(self.map_valid),
            map_polyline_types=move(self.map_polyline_types),
            map_is_intersection=move(self.map_is_intersection),
            polyline_mask=move(self.polyline_mask),
        )
