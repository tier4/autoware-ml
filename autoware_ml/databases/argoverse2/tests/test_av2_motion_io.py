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

"""Tests for the readers of the Argoverse 2 Motion Forecasting files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from autoware_ml.databases.argoverse2.av2_motion_io import (
    interp_arc,
    load_scenario_parquet,
    load_static_map_json,
)

START_NS = 315_967_000_000_000_000
STEP_NS = 100_000_000


def _write_scenario(path: Path, rows: list[dict[str, object]], num_steps: int = 3) -> None:
    """Write a scenario parquet file in the layout of the dataset."""
    frame = pl.DataFrame(rows).with_columns(
        scenario_id=pl.lit("scene"),
        start_timestamp=pl.lit(float(START_NS)),
        end_timestamp=pl.lit(float(START_NS + STEP_NS * (num_steps - 1))),
        num_timestamps=pl.lit(num_steps),
        focal_track_id=pl.lit("b"),
        city=pl.lit("miami"),
    )
    frame.write_parquet(path)


def _state(track_id: str, timestep: int, object_type: str = "vehicle", category: int = 2):
    """One row of a scenario file, its values derived from the track and the step."""
    return {
        "observed": timestep < 2,
        "track_id": track_id,
        "object_type": object_type,
        "object_category": category,
        "timestep": timestep,
        "position_x": float(timestep),
        "position_y": 1.0,
        "heading": 0.5,
        "velocity_x": 10.0,
        "velocity_y": 0.0,
    }


class TestLoadScenarioParquet:
    def test_reads_metadata_and_timestamps(self, tmp_path: Path) -> None:
        path = tmp_path / "scenario_scene.parquet"
        _write_scenario(path, [_state("b", 0), _state("b", 1), _state("b", 2)])

        scenario = load_scenario_parquet(path)

        assert scenario.scenario_id == "scene"
        assert scenario.city_name == "miami"
        assert scenario.focal_track_id == "b"
        assert scenario.timestamps_ns.dtype == np.int64
        np.testing.assert_array_equal(
            scenario.timestamps_ns, START_NS + STEP_NS * np.arange(3, dtype=np.int64)
        )

    def test_groups_states_by_track_ordered_by_track_id(self, tmp_path: Path) -> None:
        path = tmp_path / "scenario_scene.parquet"
        rows = [
            _state("b", 2, "pedestrian", 3),
            _state("a", 0),
            _state("b", 0, "pedestrian", 3),
            _state("a", 1),
        ]
        _write_scenario(path, rows)

        scenario = load_scenario_parquet(path)

        assert [track.track_id for track in scenario.tracks] == ["a", "b"]
        track_a, track_b = scenario.tracks
        assert (track_a.object_type, track_a.category) == ("vehicle", 2)
        assert (track_b.object_type, track_b.category) == ("pedestrian", 3)
        # The states of a track keep the order of the file
        np.testing.assert_array_equal(track_b.timesteps, [2, 0])
        np.testing.assert_array_equal(track_b.observed, [False, True])
        np.testing.assert_array_equal(track_b.positions, [[2.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_equal(track_b.headings, [0.5, 0.5])
        np.testing.assert_array_equal(track_b.velocities, [[10.0, 0.0], [10.0, 0.0]])

    def test_rejects_an_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scenario_scene.parquet"
        _write_scenario(path, [_state("a", 0)])
        pl.read_parquet(path).clear().write_parquet(path)

        with pytest.raises(ValueError, match="holds no track state"):
            load_scenario_parquet(path)


def _xyz(*points: tuple[float, float, float]) -> list[dict[str, float]]:
    return [{"x": x, "y": y, "z": z} for x, y, z in points]


class TestLoadStaticMapJson:
    @staticmethod
    def _write(path: Path, with_crossings: bool = True) -> None:
        data = {
            "lane_segments": {
                "7": {
                    "id": 7,
                    "is_intersection": True,
                    "lane_type": "VEHICLE",
                    "left_lane_boundary": _xyz((0, 1, 0), (5, 1, 0)),
                    "right_lane_boundary": _xyz((0, -1, 0), (2, -1, 0), (5, -1, 0)),
                    "centerline": _xyz((0, 0, 0), (5, 0, 0)),
                }
            },
            "drivable_areas": {
                "3": {"id": 3, "area_boundary": _xyz((0, 0, 0), (1, 0, 0), (1, 1, 0))}
            },
        }
        if with_crossings:
            data["pedestrian_crossings"] = {
                "9": {
                    "id": 9,
                    "edge1": _xyz((0, 0, 0), (0, 3, 0)),
                    "edge2": _xyz((1, 0, 0), (1, 3, 0)),
                }
            }
        path.write_text(json.dumps(data))

    def test_reads_every_layer(self, tmp_path: Path) -> None:
        path = tmp_path / "log_map_archive_scene.json"
        self._write(path)

        static_map = load_static_map_json(path)

        lane = static_map.lane_segments[7]
        np.testing.assert_array_equal(lane.left_boundary, [[0, 1, 0], [5, 1, 0]])
        np.testing.assert_array_equal(lane.right_boundary, [[0, -1, 0], [2, -1, 0], [5, -1, 0]])
        assert lane.is_intersection is True
        crossing = static_map.pedestrian_crossings[9]
        np.testing.assert_array_equal(crossing.edge1, [[0, 0, 0], [0, 3, 0]])
        np.testing.assert_array_equal(crossing.edge2, [[1, 0, 0], [1, 3, 0]])

    def test_closes_drivable_area_boundaries(self, tmp_path: Path) -> None:
        path = tmp_path / "log_map_archive_scene.json"
        self._write(path)

        static_map = load_static_map_json(path)

        np.testing.assert_array_equal(
            static_map.drivable_areas[3], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]
        )

    def test_map_without_crossing_layer_has_no_crossing(self, tmp_path: Path) -> None:
        path = tmp_path / "log_map_archive_scene.json"
        self._write(path, with_crossings=False)

        assert load_static_map_json(path).pedestrian_crossings == {}


class TestInterpArc:
    def test_spreads_points_evenly_along_the_length(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 3.0, 0.0]])

        resampled = interp_arc(5, points)

        np.testing.assert_allclose(
            resampled, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0]], atol=1e-12
        )

    def test_keeps_the_end_points(self) -> None:
        points = np.array([[0.0, 0.0], [2.0, 1.0], [3.0, 5.0]])

        resampled = interp_arc(7, points)

        np.testing.assert_allclose(resampled[[0, -1]], points[[0, -1]], atol=1e-12)

    def test_rejects_a_flat_array(self) -> None:
        with pytest.raises(ValueError, match="polyline needs"):
            interp_arc(3, np.zeros(3))
