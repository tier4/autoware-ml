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

"""Tests for the loading transforms of the prediction task."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_ml.dataclasses.batch.prediction import PredictionSourceSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (
    FOCAL_TRACK_ID,
    av2_prediction_taxonomy,
    write_scenario,
)
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.prediction.loading import LoadAgentTracks, LoadVectorMap
from autoware_ml.types.dataset import DatasetFormat
from autoware_ml.types.motion import PolylineType


def _source(scenario_path: Path, with_map: bool = True) -> PredictionSourceSample:
    scenario_id = scenario_path.name
    return PredictionSourceSample(
        dataset_format=DatasetFormat.AV2,
        tracks_path=str(scenario_path / f"scenario_{scenario_id}.parquet"),
        map_path=str(scenario_path / f"log_map_archive_{scenario_id}.json") if with_map else None,
        scenario_key=None,
        focal_track_id=FOCAL_TRACK_ID,
        first_step_index=0,
        current_step_index=49,
        num_steps=110,
        step_seconds=0.1,
        num_tracks=3,
    )


def _sample(source: PredictionSourceSample | None) -> ModelGTSample:
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
        prediction_source_sample=source,
    )


@pytest.fixture
def scenario_path(tmp_path: Path) -> Path:
    pytest.importorskip("av2")
    return write_scenario(tmp_path / "train", "abc")


def test_load_agent_tracks_requires_the_source_sample() -> None:
    with pytest.raises(KeyError, match="prediction_source_sample"):
        LoadAgentTracks(av2_prediction_taxonomy())(_sample(None))


def test_load_agent_tracks_requires_a_label_taxonomy() -> None:
    with pytest.raises(TypeError, match="must be a LabelTaxonomy"):
        LoadAgentTracks(av2_prediction_taxonomy().vocabulary)


def test_load_agent_tracks_fills_the_gt_sample(scenario_path: Path) -> None:
    loaded = LoadAgentTracks(av2_prediction_taxonomy())(_sample(_source(scenario_path)))

    gt_sample = loaded.prediction_gt_sample
    assert gt_sample is not None
    assert gt_sample.map_polylines is None
    assert gt_sample.agents.positions.shape == (3, 110, 3)
    assert gt_sample.agents.track_ids[0] == FOCAL_TRACK_ID
    assert gt_sample.agents.current_step_index == 49
    assert loaded.prediction_source_sample is not None


def test_load_vector_map_requires_the_tracks_first(scenario_path: Path) -> None:
    with pytest.raises(KeyError, match="prediction_gt_sample"):
        LoadVectorMap()(_sample(_source(scenario_path)))


def test_load_vector_map_rejects_a_scenario_without_a_map(scenario_path: Path) -> None:
    loaded = LoadAgentTracks(av2_prediction_taxonomy())(
        _sample(_source(scenario_path, with_map=False))
    )

    with pytest.raises(ValueError, match="has no map"):
        LoadVectorMap()(loaded)


def test_load_vector_map_rejects_too_few_points_and_unknown_types() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        LoadVectorMap(num_points_per_polyline=1)
    with pytest.raises(ValueError, match="Unknown polyline type 'river'"):
        LoadVectorMap(polyline_types=["river"])


def test_the_composed_pipeline_loads_tracks_and_map(scenario_path: Path) -> None:
    pipeline = TransformsCompose(
        [
            LoadAgentTracks(av2_prediction_taxonomy()),
            LoadVectorMap(num_points_per_polyline=5, polyline_types=["lane_centerline", 3]),
        ]
    )

    loaded = pipeline(_sample(_source(scenario_path)))

    gt_sample = loaded.prediction_gt_sample
    assert gt_sample.agents.num_agents == 3
    assert gt_sample.map_polylines is not None
    assert gt_sample.map_polylines.points.shape == (3, 5, 3)
    assert gt_sample.map_polylines.polyline_types.tolist() == [
        PolylineType.LANE_CENTERLINE,
        PolylineType.CROSSWALK_EDGE,
        PolylineType.CROSSWALK_EDGE,
    ]
    assert "LANE_CENTERLINE" in repr(pipeline)


def test_reloading_the_tracks_keeps_a_loaded_map(scenario_path: Path) -> None:
    load_tracks = LoadAgentTracks(av2_prediction_taxonomy())
    loaded = LoadVectorMap()(load_tracks(_sample(_source(scenario_path))))

    reloaded = load_tracks(loaded)

    assert reloaded.prediction_gt_sample.map_polylines is loaded.prediction_gt_sample.map_polylines
