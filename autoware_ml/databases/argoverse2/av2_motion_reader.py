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

"""Readers of the Argoverse 2 Motion Forecasting files.

This is the only module that imports the ``av2`` devkit. A scenario is a
``scenario_<id>.parquet`` file of agent tracks and a ``log_map_archive_<id>.json`` static map;
the readers turn them into the dataset agnostic samples of the prediction task, in the city
frame of the scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, Track
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.geometry.interpolate import interp_arc
from av2.map.map_api import ArgoverseStaticMap
from jaxtyping import Float64

from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.dataclasses.batch.prediction import AgentTracksSample, MapPolylinesSample
from autoware_ml.types.motion import AgentRole, PolylineType

# Number of points every map polyline is resampled to by default.
DEFAULT_NUM_POINTS_PER_POLYLINE = 20


class ScenarioMetadata(NamedTuple):
    """
    Named tuple with the metadata of one scenario, read when the records are generated.

    Attributes:
      scenario_id: Identifier of the scenario.
      city: City the scenario was recorded in.
      focal_track_id: Identifier of the focal agent.
      timestamps_seconds: Timestamp of every step of the window in seconds.
      num_tracks: Number of agent tracks in the scenario.
    """

    scenario_id: str
    city: str
    focal_track_id: str
    timestamps_seconds: Float64[np.ndarray, " num_steps"]
    num_tracks: int

    @property
    def num_steps(self) -> int:
        """Number of steps in the window."""
        return len(self.timestamps_seconds)

    @property
    def step_seconds(self) -> float:
        """Period between two steps in seconds."""
        return step_seconds_of(self.timestamps_seconds)


def step_seconds_of(timestamps_seconds: Float64[np.ndarray, " num_steps"]) -> float:
    """
    Period of a uniformly sampled time axis.

    Args:
      timestamps_seconds: Timestamp of every step in seconds, at least two of them.

    Returns:
      float: The period between two steps in seconds.

    Raises:
      ValueError: If there are fewer than two steps or the steps are not uniform.
    """

    if len(timestamps_seconds) < 2:
        raise ValueError(
            f"A scenario needs at least two steps to have a period, got {len(timestamps_seconds)}."
        )
    periods = np.diff(timestamps_seconds)
    step_seconds = float(np.median(periods))
    if step_seconds <= 0.0 or not np.allclose(periods, step_seconds, rtol=1e-3, atol=1e-6):
        raise ValueError(
            "The steps of the scenario are not uniformly spaced, got periods between "
            f"{periods.min()} s and {periods.max()} s."
        )
    return step_seconds


def read_scenario_metadata(tracks_path: Path) -> ScenarioMetadata:
    """
    Read the metadata of a scenario.

    Args:
      tracks_path: Path of the ``scenario_<id>.parquet`` file.

    Returns:
      ScenarioMetadata: The metadata of the scenario.
    """

    scenario = load_argoverse_scenario_parquet(Path(tracks_path))
    return ScenarioMetadata(
        scenario_id=scenario.scenario_id,
        city=scenario.city_name,
        focal_track_id=scenario.focal_track_id,
        timestamps_seconds=np.asarray(scenario.timestamps_ns, dtype=np.float64) * 1e-9,
        num_tracks=len(scenario.tracks),
    )


def read_agent_tracks(
    tracks_path: Path, taxonomy: LabelTaxonomy, current_step_index: int
) -> AgentTracksSample:
    """
    Read the agent tracks of a scenario.

    Args:
      tracks_path: Path of the ``scenario_<id>.parquet`` file.
      taxonomy: Taxonomy the agent types are resolved with; an unlisted object type raises.
      current_step_index: Index of the current step inside the window of the scenario.

    Returns:
      AgentTracksSample: The tracks, the focal agent first, in the city frame. ``sizes`` is
        None: the Argoverse 2 object states record position, heading and velocity only.
    """

    scenario = load_argoverse_scenario_parquet(Path(tracks_path))
    num_steps = len(scenario.timestamps_ns)
    if not 0 <= current_step_index < num_steps:
        raise ValueError(
            f"current_step_index must lie in [0, {num_steps}) for scenario "
            f"{scenario.scenario_id}, got {current_step_index}."
        )
    tracks = _focal_first(scenario)
    num_agents = len(tracks)

    positions = np.zeros((num_agents, num_steps, 3), dtype=np.float32)
    headings = np.zeros((num_agents, num_steps), dtype=np.float32)
    velocities = np.zeros((num_agents, num_steps, 2), dtype=np.float32)
    valid = np.zeros((num_agents, num_steps), dtype=bool)
    agent_types = np.empty(num_agents, dtype=np.int64)
    agent_roles = np.empty(num_agents, dtype=np.int64)
    for agent_index, track in enumerate(tracks):
        agent_types[agent_index] = taxonomy.resolve_index(track.object_type.value)
        agent_roles[agent_index] = AgentRole(track.category.value)
        for state in track.object_states:
            if not 0 <= state.timestep < num_steps:
                raise ValueError(
                    f"Track {track.track_id} of scenario {scenario.scenario_id} has a state at "
                    f"step {state.timestep}, outside the window of {num_steps} steps."
                )
            positions[agent_index, state.timestep, :2] = state.position
            headings[agent_index, state.timestep] = state.heading
            velocities[agent_index, state.timestep] = state.velocity
            valid[agent_index, state.timestep] = True

    return AgentTracksSample(
        positions=torch.from_numpy(positions),
        headings=torch.from_numpy(headings),
        velocities=torch.from_numpy(velocities),
        valid=torch.from_numpy(valid),
        agent_types=torch.from_numpy(agent_types),
        agent_roles=torch.from_numpy(agent_roles),
        track_ids=[track.track_id for track in tracks],
        current_step_index=current_step_index,
        step_seconds=step_seconds_of(np.asarray(scenario.timestamps_ns, dtype=np.float64) * 1e-9),
        ignore_index=taxonomy.ignore_index,
        sizes=None,
    )


def _focal_first(scenario: ArgoverseScenario) -> list[Track]:
    """
    Order the tracks of a scenario with the focal track first.

    Args:
      scenario: The scenario.

    Returns:
      list[Track]: The tracks, the focal one first and the others in file order.

    Raises:
      ValueError: If the focal track is missing or listed more than once.
    """

    focal = [track for track in scenario.tracks if track.track_id == scenario.focal_track_id]
    if len(focal) != 1:
        raise ValueError(
            f"Scenario {scenario.scenario_id} names the focal track "
            f"{scenario.focal_track_id!r}, which appears {len(focal)} times in its tracks."
        )
    others = [track for track in scenario.tracks if track.track_id != scenario.focal_track_id]
    return focal + others


def read_vector_map(
    map_path: Path,
    num_points_per_polyline: int = DEFAULT_NUM_POINTS_PER_POLYLINE,
    polyline_types: Sequence[PolylineType] = tuple(PolylineType),
) -> MapPolylinesSample:
    """
    Read the static map of a scenario as polylines of a fixed number of points.

    A lane segment yields its centerline and its two boundaries, a pedestrian crossing its two
    edges, and a drivable area its closed exterior boundary. Every polyline is resampled to
    ``num_points_per_polyline`` points at equal arc length. Lane connectivity is not exposed.

    Args:
      map_path: Path of the ``log_map_archive_<id>.json`` file; the devkit parses the log id
        from the file name, so the file keeps its original name.
      num_points_per_polyline: Number of points every polyline is resampled to.
      polyline_types: Kinds of polylines to keep, in the order of the enum.

    Returns:
      MapPolylinesSample: The polylines, lane segments first, then crossings, then drivable
        areas, each group in element id order.
    """

    if num_points_per_polyline < 2:
        raise ValueError(
            f"num_points_per_polyline must be at least 2, got {num_points_per_polyline}."
        )
    kept_types = set(polyline_types)
    static_map = ArgoverseStaticMap.from_json(Path(map_path))

    points: list[np.ndarray] = []
    types: list[int] = []
    is_intersection: list[bool] = []
    element_ids: list[str] = []

    def add(polyline: np.ndarray, kind: PolylineType, element_id: int, intersection: bool) -> None:
        if kind not in kept_types:
            return
        points.append(_resample(np.asarray(polyline, dtype=np.float64), num_points_per_polyline))
        types.append(int(kind))
        is_intersection.append(intersection)
        element_ids.append(str(element_id))

    for lane_id in sorted(static_map.vector_lane_segments):
        lane = static_map.vector_lane_segments[lane_id]
        left = _resample(
            np.asarray(lane.left_lane_boundary.xyz, dtype=np.float64), num_points_per_polyline
        )
        right = _resample(
            np.asarray(lane.right_lane_boundary.xyz, dtype=np.float64), num_points_per_polyline
        )
        add((left + right) / 2.0, PolylineType.LANE_CENTERLINE, lane_id, lane.is_intersection)
        add(left, PolylineType.LANE_LEFT_BOUNDARY, lane_id, lane.is_intersection)
        add(right, PolylineType.LANE_RIGHT_BOUNDARY, lane_id, lane.is_intersection)
    for crossing_id in sorted(static_map.vector_pedestrian_crossings):
        crossing = static_map.vector_pedestrian_crossings[crossing_id]
        add(crossing.edge1.xyz, PolylineType.CROSSWALK_EDGE, crossing_id, False)
        add(crossing.edge2.xyz, PolylineType.CROSSWALK_EDGE, crossing_id, False)
    for area_id in sorted(static_map.vector_drivable_areas):
        area = static_map.vector_drivable_areas[area_id]
        add(area.xyz, PolylineType.DRIVABLE_AREA_BOUNDARY, area_id, False)

    num_polylines = len(points)
    stacked = (
        np.stack(points).astype(np.float32)
        if num_polylines
        else np.zeros((0, num_points_per_polyline, 3), dtype=np.float32)
    )
    return MapPolylinesSample(
        points=torch.from_numpy(stacked),
        valid=torch.ones((num_polylines, num_points_per_polyline), dtype=torch.bool),
        polyline_types=torch.tensor(types, dtype=torch.int64),
        is_intersection=torch.tensor(is_intersection, dtype=torch.bool),
        element_ids=element_ids,
    )


def _resample(polyline: Float64[np.ndarray, "num_points 3"], num_points: int) -> np.ndarray:
    """
    Resample a polyline to a fixed number of points at equal arc length.

    Args:
      polyline: Points of the polyline, at least one.
      num_points: Number of points to resample to.

    Returns:
      np.ndarray: The resampled polyline of shape (num_points, 3).
    """

    if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) == 0:
        raise ValueError(f"A polyline needs a non-empty (N, 3) array, got shape {polyline.shape}.")
    # Consecutive duplicates have zero chord length and would divide by zero
    keep = np.ones(len(polyline), dtype=bool)
    keep[1:] = np.any(polyline[1:] != polyline[:-1], axis=1)
    polyline = polyline[keep]
    if len(polyline) == 1:
        return np.repeat(polyline, num_points, axis=0)
    return interp_arc(num_points, points=polyline)
