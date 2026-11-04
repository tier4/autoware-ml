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

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("av2")

from autoware_ml.databases.argoverse2.av2_motion_reader import (  # noqa: E402
    read_agent_tracks,
    read_scenario_metadata,
    read_vector_map,
    step_seconds_of,
)
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (  # noqa: E402
    CITY,
    FOCAL_TRACK_ID,
    av2_prediction_taxonomy,
    write_scenario,
)
from autoware_ml.types.motion import AgentRole, PolylineType  # noqa: E402


@pytest.fixture
def scenario_path(tmp_path: Path) -> Path:
    return write_scenario(tmp_path / "train", "abc")


def _tracks_file(scenario_path: Path) -> Path:
    return scenario_path / f"scenario_{scenario_path.name}.parquet"


def _map_file(scenario_path: Path) -> Path:
    return scenario_path / f"log_map_archive_{scenario_path.name}.json"


def test_scenario_metadata(scenario_path: Path) -> None:
    metadata = read_scenario_metadata(_tracks_file(scenario_path))

    assert metadata.scenario_id == "abc"
    assert metadata.city == CITY
    assert metadata.focal_track_id == FOCAL_TRACK_ID
    assert metadata.num_steps == 110
    assert metadata.num_tracks == 3
    assert metadata.step_seconds == pytest.approx(0.1)
    assert metadata.timestamps_seconds[0] == pytest.approx(1.0)
    assert metadata.timestamps_seconds[49] == pytest.approx(5.9)


def test_step_seconds_rejects_a_non_uniform_axis() -> None:
    with pytest.raises(ValueError, match="not uniformly spaced"):
        step_seconds_of(np.array([0.0, 0.1, 0.3]))
    with pytest.raises(ValueError, match="at least two steps"):
        step_seconds_of(np.array([0.0]))


def test_agent_tracks_shapes_and_focal_first(scenario_path: Path) -> None:
    tracks = read_agent_tracks(_tracks_file(scenario_path), av2_prediction_taxonomy(), 49)

    assert tracks.num_agents == 3
    assert tracks.num_steps == 110
    assert tracks.positions.shape == (3, 110, 3)
    assert tracks.headings.shape == (3, 110)
    assert tracks.velocities.shape == (3, 110, 2)
    assert tracks.valid.shape == (3, 110)
    assert tracks.positions.dtype == torch.float32
    assert tracks.agent_types.dtype == torch.int64
    assert tracks.agent_roles.dtype == torch.int64
    assert tracks.valid.dtype == torch.bool
    assert tracks.track_ids[0] == FOCAL_TRACK_ID
    assert set(tracks.track_ids) == {FOCAL_TRACK_ID, "walker", "cone"}
    assert tracks.current_step_index == 49
    assert tracks.step_seconds == pytest.approx(0.1)
    assert tracks.ignore_index == -1
    assert tracks.sizes is None  # Argoverse 2 records no box size


def test_agent_tracks_values(scenario_path: Path) -> None:
    tracks = read_agent_tracks(_tracks_file(scenario_path), av2_prediction_taxonomy(), 49)
    walker = tracks.track_ids.index("walker")
    cone = tracks.track_ids.index("cone")

    # The focal track moves along x at one meter per step with z = 0
    assert tracks.valid[0].all()
    assert tracks.positions[0, 7].tolist() == [7.0, 0.0, 0.0]
    assert tracks.velocities[0, 7].tolist() == [10.0, 0.0]
    assert tracks.headings[0, 7] == 0.0
    # The walker is observed at the even steps only, the odd steps are invalid zeros
    assert tracks.valid[walker, ::2].all()
    assert not tracks.valid[walker, 1::2].any()
    assert tracks.positions[walker, 1].tolist() == [0.0, 0.0, 0.0]
    assert tracks.positions[walker, 8].tolist() == [0.0, 8.0, 0.0]
    assert tracks.headings[walker, 8] == pytest.approx(np.pi / 2)
    # The cone is a fragment present at the first step only
    assert tracks.valid[cone].sum() == 1
    # Types at the coarse level, the static cone keeps the ignore index as context
    assert tracks.agent_types.tolist()[0] == 0
    assert tracks.agent_types[walker] == 1
    assert tracks.agent_types[cone] == -1
    assert tracks.agent_roles[0] == AgentRole.FOCAL
    assert tracks.agent_roles[walker] == AgentRole.SCORED
    assert tracks.agent_roles[cone] == AgentRole.TRACK_FRAGMENT


def test_agent_tracks_of_a_test_scenario(tmp_path: Path) -> None:
    scenario_path = write_scenario(tmp_path / "test", "t00", num_steps=50)

    tracks = read_agent_tracks(_tracks_file(scenario_path), av2_prediction_taxonomy(), 49)

    assert tracks.num_steps == 50
    assert tracks.valid[0].all()


def test_agent_tracks_rejects_a_current_step_outside_the_window(scenario_path: Path) -> None:
    with pytest.raises(ValueError, match="current_step_index must lie in \\[0, 110\\)"):
        read_agent_tracks(_tracks_file(scenario_path), av2_prediction_taxonomy(), 110)


def test_agent_tracks_rejects_an_unlisted_object_type(tmp_path: Path) -> None:
    scenario_path = write_scenario(tmp_path / "train", "odd", object_types={FOCAL_TRACK_ID: "bus"})
    taxonomy = av2_prediction_taxonomy()
    vocabulary = {name: fine for name, fine in taxonomy.vocabulary.class_renaming.items()}
    vocabulary.pop("bus")
    from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

    narrowed = LabelTaxonomy(
        vocabulary=LabelVocabulary(vocabulary),
        class_names=taxonomy.class_names,
        class_mapping={k: v for k, v in taxonomy.class_mapping.items() if k != "bus"},
        ignore_index=-1,
    )

    with pytest.raises(KeyError, match="'bus' is not listed in the vocabulary"):
        read_agent_tracks(_tracks_file(scenario_path), narrowed, 49)


def test_vector_map_polylines(scenario_path: Path) -> None:
    polylines = read_vector_map(_map_file(scenario_path), num_points_per_polyline=5)

    # 3 per lane, 2 per crossing, 1 per drivable area
    assert polylines.num_polylines == 6
    assert polylines.num_points_per_polyline == 5
    assert polylines.points.shape == (6, 5, 3)
    assert polylines.points.dtype == torch.float32
    assert polylines.valid.all()
    assert polylines.polyline_types.tolist() == [
        PolylineType.LANE_CENTERLINE,
        PolylineType.LANE_LEFT_BOUNDARY,
        PolylineType.LANE_RIGHT_BOUNDARY,
        PolylineType.CROSSWALK_EDGE,
        PolylineType.CROSSWALK_EDGE,
        PolylineType.DRIVABLE_AREA_BOUNDARY,
    ]
    assert polylines.element_ids == ["10", "10", "10", "20", "20", "30"]
    assert polylines.is_intersection.tolist() == [True, True, True, False, False, False]
    # The centerline is the mean of the resampled boundaries, equally spaced along x
    centerline = polylines.points[0]
    assert centerline[:, 0].tolist() == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])
    assert centerline[:, 1].tolist() == pytest.approx([0.0] * 5)
    assert polylines.points[1][:, 1].tolist() == pytest.approx([2.0] * 5)
    assert polylines.points[2][:, 1].tolist() == pytest.approx([-2.0] * 5)
    # The crossing edge runs along y
    assert polylines.points[3][:, 1].tolist() == pytest.approx([-5.0, -2.5, 0.0, 2.5, 5.0])
    # The drivable area boundary is closed: it ends where it starts
    assert polylines.points[5][0].tolist() == polylines.points[5][-1].tolist()


def test_vector_map_filters_polyline_types(scenario_path: Path) -> None:
    polylines = read_vector_map(
        _map_file(scenario_path),
        num_points_per_polyline=4,
        polyline_types=[PolylineType.LANE_CENTERLINE, PolylineType.CROSSWALK_EDGE],
    )

    assert polylines.polyline_types.tolist() == [0, 3, 3]
    assert polylines.points.shape == (3, 4, 3)


def test_vector_map_without_kept_polylines_is_empty(scenario_path: Path) -> None:
    polylines = read_vector_map(_map_file(scenario_path), polyline_types=[])

    assert polylines.num_polylines == 0
    assert polylines.points.shape == (0, 20, 3)
    assert polylines.valid.shape == (0, 20)
    assert polylines.polyline_types.shape == (0,)
    assert polylines.element_ids == []


def test_vector_map_rejects_too_few_points(scenario_path: Path) -> None:
    with pytest.raises(ValueError, match="num_points_per_polyline must be at least 2"):
        read_vector_map(_map_file(scenario_path), num_points_per_polyline=1)
