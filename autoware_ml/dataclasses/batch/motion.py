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

from collections.abc import Sequence
from typing import NamedTuple

import torch
from jaxtyping import Bool, Float32, Int64, Shaped
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

    # (A, T, 3), (x, y, z) position of every agent at every step; z is zero for a source that
    # records planar positions only (Argoverse 2)
    positions: Float32[Tensor, "num_agents num_steps 3"]
    # (A, T), heading of every agent at every step in radians
    headings: Float32[Tensor, "num_agents num_steps"]
    # (A, T, 2), planar (vx, vy) velocity of every agent at every step in meters per second
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


class MotionSourceSample(NamedTuple):
    """
    Named tuple to represent the source of one scenario, its file paths resolved against the
    root of the dataset serving it: the analog of LiDARPointCloudSample for the motion
    task. The loading transforms turn it into a MotionGTSample.
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


class MotionGTSample(NamedTuple):
    """Named tuple to represent the ground truth of the motion task for one scenario."""

    # The agent tracks over the window of the scenario
    agents: AgentTracksSample
    # The vector map of the scenario, None when it was not loaded or the scenario has none
    map_polylines: MapPolylinesSample | None = None


def _pad_rows(
    tensor: Shaped[Tensor, " num_input_rows *rest"],
    num_rows: int,
    fill: float | int | bool,
) -> Shaped[Tensor, " num_rows *rest"]:
    """
    Pad the first dimension of a tensor up to a row count.

    Args:
      tensor: Tensor to pad, of any dtype; the trailing dimensions are kept.
      num_rows: Row count to pad to, at least the rows of the tensor.
      fill: Value of the padded rows.

    Returns:
      Shaped[Tensor, " num_rows *rest"]: The padded tensor, of the dtype of the input.
    """

    padding = num_rows - tensor.shape[0]
    if padding == 0:
        return tensor
    pad = torch.full((padding, *tensor.shape[1:]), fill, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _stack_padded(
    tensors: Sequence[Shaped[Tensor, " num_input_rows *rest"]],
    num_rows: int,
    fill: float | int | bool,
) -> Shaped[Tensor, " batch_size num_rows *rest"]:
    """
    Pad the tensors of the samples of a batch to a shared row count and stack them.

    Args:
      tensors: One tensor per sample, sharing their trailing dimensions.
      num_rows: Row count to pad to, at least the rows of every tensor.
      fill: Value of the padded rows.

    Returns:
      Shaped[Tensor, " batch_size num_rows *rest"]: The padded tensors stacked along a new
        first dimension.
    """

    return torch.stack([_pad_rows(tensor, num_rows, fill) for tensor in tensors])


def _slot_mask(num_real_rows: int, num_rows: int) -> Bool[Tensor, " num_rows"]:
    """
    Mask of the slots of one sample after padding: the real rows first, then the padding.

    Args:
      num_real_rows: Number of rows the sample holds.
      num_rows: Row count the sample is padded to.

    Returns:
      Bool[Tensor, " num_rows"]: True for a real row, False for a padded one.
    """

    return _pad_rows(torch.ones(num_real_rows, dtype=torch.bool), num_rows, False)


class AgentTracksBatch(NamedTuple):
    """
    Named tuple to represent the agent tracks of a batch of scenarios, the batch of
    AgentTracksSample.

    The agents are padded to the largest agent count of the batch, and ``mask`` names the real
    slots. A padded agent carries zeros, an invalid track, the ignore index as type and
    PADDED_INDEX as role. Every scenario of a batch shares the window shape, so it is stored
    once.
    """

    # (B, A, T, 3), position of every agent at every step
    positions: Float32[Tensor, "batch_size num_agents num_steps 3"]
    # (B, A, T), heading of every agent at every step in radians
    headings: Float32[Tensor, "batch_size num_agents num_steps"]
    # (B, A, T, 2), planar velocity of every agent at every step
    velocities: Float32[Tensor, "batch_size num_agents num_steps 2"]
    # (B, A, T), whether the agent was observed at the step
    valid: Bool[Tensor, "batch_size num_agents num_steps"]
    # (B, A), class index of every agent, the ignore index for a dropped or padded agent
    agent_types: Int64[Tensor, "batch_size num_agents"]
    # (B, A), AgentRole value of every agent, PADDED_INDEX for a padded slot
    agent_roles: Int64[Tensor, "batch_size num_agents"]
    # (B, A), whether the agent slot holds an agent of the scenario
    mask: Bool[Tensor, "batch_size num_agents"]
    # Source identifier of the agents of every scenario, the real slots only
    track_ids: Sequence[Sequence[str]]
    # Index of the current step, shared by every scenario of the batch
    current_step_index: int
    # Period between two steps in seconds, shared by every scenario of the batch
    step_seconds: float
    # Class index of an agent the taxonomy does not name
    ignore_index: int
    # (B, A, 3), size of every agent, None when the samples carry no sizes
    sizes: Float32[Tensor, "batch_size num_agents 3"] | None = None

    @property
    def batch_size(self) -> int:
        """Number of collated scenarios."""
        return int(self.mask.shape[0])

    @staticmethod
    def _check_shared_settings(samples: Sequence[AgentTracksSample]) -> None:
        """
        Check that the samples share the window shape, the taxonomy and the presence of sizes.

        Args:
          samples: Agent tracks of the scenarios to collate, at least one.

        Raises:
          ValueError: If a sample disagrees with the first one.
        """

        first = samples[0]
        first_window = (first.num_steps, first.current_step_index, first.step_seconds)
        for index, sample in enumerate(samples):
            window = (sample.num_steps, sample.current_step_index, sample.step_seconds)
            if window != first_window:
                raise ValueError(
                    f"Sample {index} has a window of {window} (num_steps, current_step_index, "
                    f"step_seconds), sample 0 has {first_window}; every scenario of a batch "
                    "shares the window shape."
                )
            if sample.ignore_index != first.ignore_index:
                raise ValueError(
                    f"Sample {index} has the ignore index {sample.ignore_index}, sample 0 has "
                    f"{first.ignore_index}; every scenario of a batch shares the taxonomy."
                )
        with_sizes = [sample.sizes is not None for sample in samples]
        if any(with_sizes) and not all(with_sizes):
            raise ValueError("Either every sample carries agent sizes or none does.")

    @staticmethod
    def collate(samples: Sequence[AgentTracksSample]) -> AgentTracksBatch:
        """
        Collate the agent tracks of a sequence of scenarios into a single AgentTracksBatch.

        Args:
          samples: Agent tracks of the scenarios to collate, at least one.

        Returns:
          AgentTracksBatch: The agent tracks padded to the largest agent count.

        Raises:
          ValueError: If the samples disagree on the window shape, the ignore index or the
            presence of sizes.
        """

        AgentTracksBatch._check_shared_settings(samples)
        first = samples[0]
        num_agents = max(sample.num_agents for sample in samples)

        return AgentTracksBatch(
            positions=_stack_padded([s.positions for s in samples], num_agents, 0.0),
            headings=_stack_padded([s.headings for s in samples], num_agents, 0.0),
            velocities=_stack_padded([s.velocities for s in samples], num_agents, 0.0),
            valid=_stack_padded([s.valid for s in samples], num_agents, False),
            agent_types=_stack_padded(
                [s.agent_types for s in samples], num_agents, first.ignore_index
            ),
            agent_roles=_stack_padded([s.agent_roles for s in samples], num_agents, PADDED_INDEX),
            mask=torch.stack([_slot_mask(s.num_agents, num_agents) for s in samples]),
            track_ids=[list(sample.track_ids) for sample in samples],
            current_step_index=first.current_step_index,
            step_seconds=first.step_seconds,
            ignore_index=first.ignore_index,
            sizes=(
                _stack_padded([s.sizes for s in samples], num_agents, 0.0)
                if first.sizes is not None
                else None
            ),
        )

    def to_device(self, device: torch.device) -> AgentTracksBatch:
        """
        Move the AgentTracksBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          AgentTracksBatch: The batch moved to the specified device.
        """

        return self._replace(
            positions=self.positions.to(device),
            headings=self.headings.to(device),
            velocities=self.velocities.to(device),
            valid=self.valid.to(device),
            agent_types=self.agent_types.to(device),
            agent_roles=self.agent_roles.to(device),
            mask=self.mask.to(device),
            sizes=self.sizes.to(device) if self.sizes is not None else None,
        )


class MapPolylinesBatch(NamedTuple):
    """
    Named tuple to represent the vector maps of a batch of scenarios, the batch of
    MapPolylinesSample.

    The polylines are padded to the largest polyline count of the batch, and ``mask`` names
    the real slots. A padded polyline carries zeros, invalid points and PADDED_INDEX as type.
    """

    # (B, P, L, 3), points of every polyline
    points: Float32[Tensor, "batch_size num_polylines num_points 3"]
    # (B, P, L), whether the point of the polyline is set
    valid: Bool[Tensor, "batch_size num_polylines num_points"]
    # (B, P), PolylineType value of every polyline, PADDED_INDEX for a padded slot
    polyline_types: Int64[Tensor, "batch_size num_polylines"]
    # (B, P), whether the polyline belongs to an intersection
    is_intersection: Bool[Tensor, "batch_size num_polylines"]
    # (B, P), whether the polyline slot holds a polyline of the scenario
    mask: Bool[Tensor, "batch_size num_polylines"]
    # Source identifier of the polylines of every scenario, the real slots only
    element_ids: Sequence[Sequence[str]]

    @staticmethod
    def collate(samples: Sequence[MapPolylinesSample]) -> MapPolylinesBatch:
        """
        Collate the vector maps of a sequence of scenarios into a single MapPolylinesBatch.

        Args:
          samples: Vector maps of the scenarios to collate, at least one.

        Returns:
          MapPolylinesBatch: The polylines padded to the largest polyline count.

        Raises:
          ValueError: If the samples disagree on the points per polyline.
        """

        num_points = samples[0].num_points_per_polyline
        for index, sample in enumerate(samples):
            if sample.num_points_per_polyline != num_points:
                raise ValueError(
                    f"Sample {index} has {sample.num_points_per_polyline} points per polyline, "
                    f"sample 0 has {num_points}; every map of a batch shares it."
                )
        num_polylines = max(sample.num_polylines for sample in samples)

        return MapPolylinesBatch(
            points=_stack_padded([s.points for s in samples], num_polylines, 0.0),
            valid=_stack_padded([s.valid for s in samples], num_polylines, False),
            polyline_types=_stack_padded(
                [s.polyline_types for s in samples], num_polylines, PADDED_INDEX
            ),
            is_intersection=_stack_padded(
                [s.is_intersection for s in samples], num_polylines, False
            ),
            mask=torch.stack([_slot_mask(s.num_polylines, num_polylines) for s in samples]),
            element_ids=[list(sample.element_ids) for sample in samples],
        )

    def to_device(self, device: torch.device) -> MapPolylinesBatch:
        """
        Move the MapPolylinesBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          MapPolylinesBatch: The batch moved to the specified device.
        """

        return self._replace(
            points=self.points.to(device),
            valid=self.valid.to(device),
            polyline_types=self.polyline_types.to(device),
            is_intersection=self.is_intersection.to(device),
            mask=self.mask.to(device),
        )


class MotionGTBatch(NamedTuple):
    """
    Named tuple to represent the ground truth of the motion task for a batch of scenarios: the
    batch of MotionGTSample.
    """

    # The agent tracks of every scenario
    agents: AgentTracksBatch
    # The vector maps of every scenario, None when the samples carry no map
    map_polylines: MapPolylinesBatch | None = None

    @property
    def batch_size(self) -> int:
        """Number of collated scenarios."""
        return self.agents.batch_size

    @property
    def has_map(self) -> bool:
        """Whether the batch carries the vector maps of its scenarios."""
        return self.map_polylines is not None

    @staticmethod
    def collate_gt_samples(
        motion_gt_samples: Sequence[MotionGTSample],
    ) -> MotionGTBatch | None:
        """
        Collate a sequence of motion GT samples into a single MotionGTBatch.

        Args:
          motion_gt_samples: Sequence of MotionGTSample to be collated.

        Returns:
          MotionGTBatch: Collated motion GT batch, None for an empty sequence.

        Raises:
          ValueError: If the samples disagree on the window shape, the ignore index, the
            presence of sizes or of a map, or the points per polyline.
        """

        if len(motion_gt_samples) == 0:
            return None

        agents = AgentTracksBatch.collate([sample.agents for sample in motion_gt_samples])

        maps = [sample.map_polylines for sample in motion_gt_samples]
        with_map = [polylines is not None for polylines in maps]
        if not any(with_map):
            return MotionGTBatch(agents=agents)
        if not all(with_map):
            raise ValueError("Either every sample carries a vector map or none does.")
        return MotionGTBatch(agents=agents, map_polylines=MapPolylinesBatch.collate(maps))

    def to_device(self, device: torch.device) -> MotionGTBatch:
        """
        Move the MotionGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          MotionGTBatch: The batch moved to the specified device.
        """

        return MotionGTBatch(
            agents=self.agents.to_device(device),
            map_polylines=(
                self.map_polylines.to_device(device) if self.map_polylines is not None else None
            ),
        )
