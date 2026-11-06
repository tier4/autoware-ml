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

"""Synthetic Argoverse 2 Motion Forecasting scenarios for the tests.

The builders write the two files of a scenario in the layout of the dataset,
``<split>/<scenario_id>/scenario_<scenario_id>.parquet`` and
``<split>/<scenario_id>/log_map_archive_<scenario_id>.json``, so every layer from the reader to
the datamodule is tested against the real file formats. The ``av2`` devkit is imported inside
the builders, so a test module can skip when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

FOCAL_TRACK_ID = "focal"
STEP_NS = 100_000_000
# Steps of the full window the scenario files store the timestamps of.
WINDOW_STEPS = 110
CITY = "PIT"

# The object types of the three tracks every synthetic scenario carries, in file order
TRACK_OBJECT_TYPES = {FOCAL_TRACK_ID: "vehicle", "walker": "pedestrian", "cone": "static"}


def av2_vocabulary() -> LabelVocabulary:
    """The vocabulary of section 3.2 of the integration plan."""
    return LabelVocabulary(
        {
            "vehicle": "vehicle",
            "bus": "bus",
            "pedestrian": "pedestrian",
            "cyclist": "cyclist",
            "motorcyclist": "motorcyclist",
            "riderless_bicycle": "riderless_bicycle",
            "static": None,
            "background": None,
            "construction": None,
            "unknown": None,
        }
    )


def av2_prediction_taxonomy() -> LabelTaxonomy:
    """The coarse prediction classes of section 3.2 of the integration plan."""
    return LabelTaxonomy(
        vocabulary=av2_vocabulary(),
        class_names=["vehicle", "pedestrian", "cyclist"],
        class_mapping={
            "vehicle": "vehicle",
            "bus": "vehicle",
            "pedestrian": "pedestrian",
            "cyclist": "cyclist",
            "motorcyclist": "cyclist",
            "riderless_bicycle": None,
        },
        ignore_index=-1,
    )


def write_scenario(
    split_path: Path,
    scenario_id: str,
    num_steps: int = 110,
    object_types: dict[str, str] | None = None,
    start_timestamp_ns: int = 1_000_000_000,
) -> Path:
    """
    Write one synthetic scenario with its tracks file and its map file.

    The focal track is observed at every step and moves along x at 1 m per step. The walker
    is observed at the even steps only and moves along y. The third track is static and present
    at the first step only.

    Args:
      split_path: The split directory, created when missing.
      scenario_id: Identifier of the scenario, also the name of its directory.
      num_steps: Number of steps the tracks carry states for, 110 for train / val and 50 for
        test; the file stores the timestamps of all 110 steps either way, as the dataset does.
      object_types: Object type of every track by track id, the default three tracks when None.
      start_timestamp_ns: Timestamp of the first step in nanoseconds.

    Returns:
      Path: The scenario directory.
    """

    from av2.datasets.motion_forecasting.data_schema import (
        ArgoverseScenario,
        ObjectState,
        ObjectType,
        Track,
        TrackCategory,
    )
    from av2.datasets.motion_forecasting.scenario_serialization import (
        serialize_argoverse_scenario_parquet,
    )

    object_types = TRACK_OBJECT_TYPES if object_types is None else object_types
    # The scenario file stores the timestamps of the full window in every split, the test
    # split included, where the tracks carry states for the observed steps only.
    timestamps_ns = start_timestamp_ns + STEP_NS * np.arange(WINDOW_STEPS, dtype=np.int64)

    def states(steps: Sequence[int], along_x: bool) -> list[ObjectState]:
        return [
            ObjectState(
                observed=step < 50,
                timestep=int(step),
                position=(float(step), 0.0) if along_x else (0.0, float(step)),
                heading=0.0 if along_x else float(np.pi / 2),
                velocity=(10.0, 0.0) if along_x else (0.0, 10.0),
            )
            for step in steps
        ]

    schedules = {
        FOCAL_TRACK_ID: (range(num_steps), True, TrackCategory.FOCAL_TRACK),
        "walker": (range(0, num_steps, 2), False, TrackCategory.SCORED_TRACK),
        "cone": ([0], True, TrackCategory.TRACK_FRAGMENT),
    }
    tracks = []
    for track_id, object_type in object_types.items():
        steps, along_x, category = schedules.get(
            track_id, (range(num_steps), True, TrackCategory.UNSCORED_TRACK)
        )
        tracks.append(
            Track(
                track_id=track_id,
                object_states=states(steps, along_x),
                object_type=ObjectType(object_type),
                category=category,
            )
        )

    scenario_path = split_path / scenario_id
    scenario_path.mkdir(parents=True, exist_ok=True)
    serialize_argoverse_scenario_parquet(
        scenario_path / f"scenario_{scenario_id}.parquet",
        ArgoverseScenario(
            scenario_id=scenario_id,
            timestamps_ns=timestamps_ns,
            tracks=tracks,
            focal_track_id=FOCAL_TRACK_ID,
            city_name=CITY,
            map_id=None,
            slice_id=None,
        ),
    )
    write_map(scenario_path / f"log_map_archive_{scenario_id}.json")
    return scenario_path


def _points(coordinates: Sequence[tuple[float, float, float]]) -> list[dict[str, float]]:
    return [{"x": x, "y": y, "z": z} for x, y, z in coordinates]


def write_map(map_path: Path) -> None:
    """
    Write a minimal static map: one lane segment, one pedestrian crossing, one drivable area.

    The lane runs along x from 0 to 100 with its left boundary at y = 2 and its right boundary
    at y = -2, so its centerline lies on y = 0.

    Args:
      map_path: Path of the ``log_map_archive_<id>.json`` file to write.
    """

    vector_map: dict[str, Any] = {
        "lane_segments": {
            "10": {
                "id": 10,
                "is_intersection": True,
                "lane_type": "VEHICLE",
                "left_lane_boundary": _points(
                    [(0.0, 2.0, 0.0), (50.0, 2.0, 0.0), (100.0, 2.0, 0.0)]
                ),
                "right_lane_boundary": _points([(0.0, -2.0, 0.0), (100.0, -2.0, 0.0)]),
                "left_lane_mark_type": "SOLID_WHITE",
                "right_lane_mark_type": "DASHED_WHITE",
                "left_neighbor_id": None,
                "right_neighbor_id": None,
                "predecessors": [],
                "successors": [],
            }
        },
        "pedestrian_crossings": {
            "20": {
                "id": 20,
                "edge1": _points([(40.0, -5.0, 0.0), (40.0, 5.0, 0.0)]),
                "edge2": _points([(44.0, -5.0, 0.0), (44.0, 5.0, 0.0)]),
            }
        },
        "drivable_areas": {
            "30": {
                "id": 30,
                "area_boundary": _points(
                    [(0.0, -3.0, 0.0), (100.0, -3.0, 0.0), (100.0, 3.0, 0.0), (0.0, 3.0, 0.0)]
                ),
            }
        },
    }
    map_path.write_text(json.dumps(vector_map))
