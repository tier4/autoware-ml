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

"""Tests for the prediction schema and its column of the record table."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.schemas.prediction import PredictionDataModel, PredictionDatasetSchema
from autoware_ml.types.dataset import DatasetFormat


def _prediction(**overrides: Any) -> PredictionDataModel:
    fields: dict[str, Any] = {
        "dataset_format": DatasetFormat.AV2,
        "tracks_relative_path": "train/abc/scenario_abc.parquet",
        "map_relative_path": "train/abc/log_map_archive_abc.json",
        "scenario_key": None,
        "focal_track_id": "12345",
        "first_step_index": 0,
        "current_step_index": 49,
        "num_steps": 110,
        "step_seconds": 0.1,
        "num_tracks": 37,
    }
    return PredictionDataModel(**(fields | overrides))


def _recording_prediction(**overrides: Any) -> PredictionDataModel:
    # A T4Dataset record: the tracks are the frame records of the window, the map comes from
    # the map provider, so the struct holds the window only
    fields: dict[str, Any] = {
        "dataset_format": DatasetFormat.T4,
        "tracks_relative_path": None,
        "map_relative_path": None,
        "scenario_key": None,
        "focal_track_id": None,
        "first_step_index": 12,
        "current_step_index": 4,
        "num_steps": 17,
        "step_seconds": 0.5,
        "num_tracks": 8,
    }
    return PredictionDataModel(**(fields | overrides))


def _record(prediction: PredictionDataModel | None) -> DatasetRecord:
    return DatasetRecord(
        scenario_id="abc",
        sample_id="abc",
        sample_index=0,
        timestamp_seconds=4.9,
        location="PIT",
        vehicle_type=None,
        scenario_name="abc",
        lidar_frames=[],
        lidar_sources=None,
        image_frames=None,
        category_mapping=None,
        boxes_3d=None,
        prediction=prediction,
    )


def _round_trip(records: list[DatasetRecord]) -> list[DatasetRecord]:
    frame = pl.DataFrame(
        [record.to_dictionary() for record in records],
        schema=DatasetTableSchema.to_polars_schema(),
    )
    return [DatasetRecord.load_from_dictionary(row) for row in frame.to_dicts()]


def test_struct_fields_match_the_data_model() -> None:
    field_names = [field.name for field in PredictionDatasetSchema.to_polars_field_schema()]

    assert field_names == list(PredictionDataModel.model_fields)
    assert set(_prediction().to_dictionary()) == set(field_names)


def test_prediction_round_trips_through_the_record_table() -> None:
    prediction = _prediction()

    (restored,) = _round_trip([_record(prediction)])

    assert restored.prediction == prediction
    assert restored.prediction.dataset_format is DatasetFormat.AV2
    assert restored.lidar_frames == []


def test_a_recording_record_round_trips() -> None:
    prediction = _recording_prediction()

    (restored,) = _round_trip([_record(prediction)])

    assert restored.prediction == prediction
    assert restored.prediction.dataset_format is DatasetFormat.T4
    assert restored.prediction.tracks_relative_path is None
    assert restored.prediction.map_relative_path is None
    assert restored.prediction.first_step_index == 12


def test_a_perception_record_stores_a_null_struct() -> None:
    # The column is nullable, so the T4 / nuScenes generators never set it and a record read
    # back from a table materializes the null struct as an absent model
    record = _record(None)

    assert record.to_dictionary()[DatasetTableSchema.PREDICTION.name] is None
    restored, with_prediction = _round_trip([record, _record(_prediction())])
    assert restored.prediction is None
    assert with_prediction.prediction == _prediction()


def test_the_record_defaults_to_no_prediction() -> None:
    record = DatasetRecord(
        scenario_id="abc",
        sample_id="abc",
        sample_index=0,
        timestamp_seconds=4.9,
        location=None,
        vehicle_type=None,
        scenario_name="abc",
        lidar_frames=[],
        lidar_sources=None,
        image_frames=None,
        category_mapping=None,
        boxes_3d=None,
    )

    assert record.prediction is None


def test_prediction_rejects_absolute_paths() -> None:
    with pytest.raises(ValueError, match="tracks_relative_path .* is absolute"):
        _prediction(tracks_relative_path="/data/av2/train/abc/scenario_abc.parquet")
    with pytest.raises(ValueError, match="map_relative_path .* is absolute"):
        _prediction(map_relative_path="/data/av2/train/abc/log_map_archive_abc.json")
    with pytest.raises(ValueError, match="tracks_relative_path is empty"):
        _prediction(tracks_relative_path="")


def test_prediction_rejects_a_malformed_window() -> None:
    with pytest.raises(ValueError, match="first_step_index must not be negative"):
        _prediction(first_step_index=-1)
    with pytest.raises(ValueError, match="current_step_index must lie in \\[0, 110\\)"):
        _prediction(current_step_index=110)
    with pytest.raises(ValueError, match="current_step_index must lie in"):
        _prediction(current_step_index=-1)
    with pytest.raises(ValueError, match="num_steps must be positive"):
        _prediction(num_steps=0, current_step_index=0)
    with pytest.raises(ValueError, match="step_seconds must be positive"):
        _prediction(step_seconds=0.0)
    with pytest.raises(ValueError, match="num_tracks must not be negative"):
        _prediction(num_tracks=-1)


def test_prediction_requires_the_source_fields_of_the_format() -> None:
    # A scenario dataset names its files and focal agent, a recording dataset names none
    with pytest.raises(ValueError, match="tracks_relative_path is required for the av2 format"):
        _prediction(tracks_relative_path=None)
    with pytest.raises(ValueError, match="map_relative_path is required for the av2 format"):
        _prediction(map_relative_path=None)
    with pytest.raises(ValueError, match="focal_track_id is required for the av2 format"):
        _prediction(focal_track_id=None)
    with pytest.raises(ValueError, match="scenario_key must be None for the av2 format"):
        _prediction(scenario_key="0")
    with pytest.raises(ValueError, match="tracks_relative_path must be None for the t4 format"):
        _recording_prediction(tracks_relative_path="scene/annotation")
    with pytest.raises(ValueError, match="map_relative_path must be None for the nuscenes"):
        _recording_prediction(dataset_format=DatasetFormat.NUSCENES, map_relative_path="map.json")


def test_prediction_is_strict_about_the_format_enum() -> None:
    # The stored string is decoded in load_from_dictionary, the model itself takes the enum
    with pytest.raises(ValueError):
        _prediction(dataset_format="av2")
    with pytest.raises(ValueError):
        PredictionDataModel.load_from_dictionary(
            _prediction().to_dictionary() | {"dataset_format": "womd"}
        )
