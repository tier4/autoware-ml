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

"""Readers of the Argoverse 2 Motion Forecasting files, without the ``av2`` devkit.

A scenario is a ``scenario_<id>.parquet`` file of agent tracks and a
``log_map_archive_<id>.json`` static map. The functions here read the parts of them the motion
task uses and return the same values as ``av2==0.3.6``
(``load_argoverse_scenario_parquet``, ``ArgoverseStaticMap.from_json``, ``interp_arc``), as
numpy arrays instead of one Python object per state or point. The devkit is not a dependency:
it requires ``opencv-python``, which clashes with ``opencv-python-headless``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from jaxtyping import Bool, Float64, Int64

logger = logging.getLogger(__name__)

# Columns of a scenario parquet file, one row per state of a track.
_TRACK_COLUMNS = (
    "track_id",
    "object_type",
    "object_category",
    "timestep",
    "observed",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
)
# Columns of a scenario parquet file that repeat the same value on every row.
_SCENARIO_COLUMNS = (
    "scenario_id",
    "city",
    "focal_track_id",
    "start_timestamp",
    "end_timestamp",
    "num_timestamps",
)


class AV2Track(NamedTuple):
    """
    Named tuple with one agent track of an Argoverse 2 scenario: its states ordered as in the
    file, at the steps where the agent was observed.
    """

    # Identifier of the track
    track_id: str
    # Object type of the track, such as "vehicle" or "pedestrian"
    object_type: str
    # Track category: 0 fragment, 1 unscored, 2 scored, 3 focal
    category: int
    # (S,), step index of every state inside the window of the scenario
    timesteps: Int64[np.ndarray, " num_states"]
    # (S,), whether the state lies in the observed part of the window
    observed: Bool[np.ndarray, " num_states"]
    # (S, 2), planar position of every state
    positions: Float64[np.ndarray, "num_states 2"]
    # (S,), heading of every state in radians
    headings: Float64[np.ndarray, " num_states"]
    # (S, 2), planar velocity of every state
    velocities: Float64[np.ndarray, "num_states 2"]


class AV2Scenario(NamedTuple):
    """Named tuple with the agent tracks and the metadata of one Argoverse 2 scenario."""

    # Identifier of the scenario
    scenario_id: str
    # City the scenario was recorded in
    city_name: str
    # Identifier of the focal track
    focal_track_id: str
    # (T,), timestamp of every step of the window in nanoseconds
    timestamps_ns: Int64[np.ndarray, " num_steps"]
    # Tracks of the scenario, ordered by track id
    tracks: list[AV2Track]


class AV2LaneSegment(NamedTuple):
    """Named tuple with the boundaries of one lane segment of an Argoverse 2 static map."""

    # (N, 3), left boundary of the lane
    left_boundary: Float64[np.ndarray, "num_left_points 3"]
    # (M, 3), right boundary of the lane
    right_boundary: Float64[np.ndarray, "num_right_points 3"]
    # Whether the lane segment belongs to an intersection
    is_intersection: bool


class AV2PedestrianCrossing(NamedTuple):
    """Named tuple with the two edges of one pedestrian crossing of an Argoverse 2 static map."""

    # (N, 3), one edge of the crossing
    edge1: Float64[np.ndarray, "num_edge1_points 3"]
    # (M, 3), the other edge of the crossing
    edge2: Float64[np.ndarray, "num_edge2_points 3"]


class AV2StaticMap(NamedTuple):
    """Named tuple with the vector layers of an Argoverse 2 static map, keyed by element id."""

    # Lane segments by lane id
    lane_segments: dict[int, AV2LaneSegment]
    # Pedestrian crossings by crossing id
    pedestrian_crossings: dict[int, AV2PedestrianCrossing]
    # (N + 1, 3) closed exterior boundary of every drivable area by area id, the first point
    # repeated at the end
    drivable_areas: dict[int, Float64[np.ndarray, "num_points 3"]]


def load_scenario_parquet(path: Path) -> AV2Scenario:
    """
    Read the agent tracks and the metadata of a scenario parquet file.

    The tracks are ordered by track id and the timestamps are spread evenly between the first and
    the last one, as the ``av2`` devkit does: the file stores only those two and the step count.

    Args:
      path: Path of the ``scenario_<id>.parquet`` file.

    Returns:
      AV2Scenario: The scenario.

    Raises:
      ValueError: If the file holds no state.
    """

    frame = pl.read_parquet(path, columns=[*_TRACK_COLUMNS, *_SCENARIO_COLUMNS])
    if frame.height == 0:
        raise ValueError(f"Scenario file {path} holds no track state.")
    first = frame.row(0, named=True)
    timestamps_ns = np.linspace(
        first["start_timestamp"], first["end_timestamp"], num=first["num_timestamps"]
    ).astype(np.int64)

    # A stable sort keeps the states of a track in file order
    frame = frame.sort("track_id", maintain_order=True)
    track_ids = frame["track_id"].to_numpy()
    starts = np.concatenate([[0], np.flatnonzero(track_ids[1:] != track_ids[:-1]) + 1])
    ends = np.append(starts[1:], frame.height)

    object_types = frame["object_type"].to_numpy()
    categories = frame["object_category"].to_numpy()
    timesteps = frame["timestep"].to_numpy().astype(np.int64)
    observed = frame["observed"].to_numpy().astype(bool)
    positions = frame.select("position_x", "position_y").to_numpy().astype(np.float64)
    headings = frame["heading"].to_numpy().astype(np.float64)
    velocities = frame.select("velocity_x", "velocity_y").to_numpy().astype(np.float64)

    tracks = [
        AV2Track(
            track_id=str(track_ids[start]),
            object_type=str(object_types[start]),
            category=int(categories[start]),
            timesteps=timesteps[start:end],
            observed=observed[start:end],
            positions=positions[start:end],
            headings=headings[start:end],
            velocities=velocities[start:end],
        )
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True)
    ]

    return AV2Scenario(
        scenario_id=str(first["scenario_id"]),
        city_name=str(first["city"]),
        focal_track_id=str(first["focal_track_id"]),
        timestamps_ns=timestamps_ns,
        tracks=tracks,
    )


def load_static_map_json(path: Path) -> AV2StaticMap:
    """
    Read the vector layers of a static map JSON file.

    Args:
      path: Path of the ``log_map_archive_<id>.json`` file.

    Returns:
      AV2StaticMap: The lane segments, pedestrian crossings and drivable areas of the map. A map
        without a pedestrian crossing layer has no crossing.
    """

    with Path(path).open("rb") as file:
        data = json.load(file)

    lane_segments = {
        int(lane["id"]): AV2LaneSegment(
            left_boundary=_points(lane["left_lane_boundary"]),
            right_boundary=_points(lane["right_lane_boundary"]),
            is_intersection=bool(lane["is_intersection"]),
        )
        for lane in data["lane_segments"].values()
    }
    if "pedestrian_crossings" not in data:
        logger.warning("Static map %s has no pedestrian crossing layer.", path)
    pedestrian_crossings = {
        int(crossing["id"]): AV2PedestrianCrossing(
            edge1=_points(crossing["edge1"]), edge2=_points(crossing["edge2"])
        )
        for crossing in data.get("pedestrian_crossings", {}).values()
    }
    drivable_areas = {}
    for area in data["drivable_areas"].values():
        boundary = _points(area["area_boundary"])
        drivable_areas[int(area["id"])] = np.concatenate([boundary, boundary[:1]])

    return AV2StaticMap(
        lane_segments=lane_segments,
        pedestrian_crossings=pedestrian_crossings,
        drivable_areas=drivable_areas,
    )


def interp_arc(num_points: int, points: Float64[np.ndarray, "n d"]) -> np.ndarray:
    """
    Resample a polyline to points spread evenly along its length, by linear interpolation.

    Ported from ``av2.geometry.interpolate.interp_arc`` of ``av2==0.3.6``
    (Copyright (c) 2022 Argo AI, MIT License), which follows the chordal parameterization of
    https://www.mathworks.com/matlabcentral/fileexchange/34874-interparc. Consecutive duplicate
    points have zero length and divide by zero, so the caller removes them.

    Args:
      num_points: Number of points to return.
      points: Points of the polyline, at least two, in 2D or 3D.

    Returns:
      np.ndarray: The resampled polyline of shape (num_points, d).

    Raises:
      ValueError: If ``points`` is not a 2D array.
    """

    if points.ndim != 2:
        raise ValueError(f"A polyline needs an (N, 2) or (N, 3) array, got shape {points.shape}.")
    num_input_points = len(points)
    targets = np.linspace(0, 1, num_points)

    chord_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    chord_lengths = chord_lengths / np.sum(chord_lengths)
    cumulative = np.zeros(len(chord_lengths) + 1)
    cumulative[1:] = np.cumsum(chord_lengths)

    # Index of the segment end every target falls before, clamped to the first and last segment
    bins = np.digitize(targets, bins=cumulative).astype(int)
    bins[(bins <= 0) | (targets <= 0)] = 1
    bins[(bins >= num_input_points) | (targets >= 1)] = num_input_points - 1

    fractions = (targets - cumulative[bins - 1]) / chord_lengths[bins - 1]
    anchors = points[bins - 1]
    return anchors + (points[bins] - anchors) * fractions[:, None]


def _points(coordinates: Sequence[Mapping[str, Any]]) -> Float64[np.ndarray, "num_points 3"]:
    """
    Convert the points of a map element to an array.

    Args:
      coordinates: Points of the element as ``{"x": ..., "y": ..., "z": ...}`` mappings.

    Returns:
      Float64[np.ndarray, "num_points 3"]: The points in order.
    """

    return np.array(
        [(point["x"], point["y"], point["z"]) for point in coordinates], dtype=np.float64
    ).reshape(-1, 3)
