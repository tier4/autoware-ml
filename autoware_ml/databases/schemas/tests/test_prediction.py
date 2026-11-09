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
import unittest

import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.schemas.prediction import PredictionDataModel, PredictionDatasetSchema
from autoware_ml.types.dataset import DatasetFormat


def _prediction(**overrides: Any) -> PredictionDataModel:
    """Build an Argoverse 2 prediction record with the given fields overridden."""
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
    """Build a T4Dataset prediction record with the given fields overridden.

    The tracks are the frame records of the window and the map comes from the map provider, so
    the struct holds the window only.
    """
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
    """Build a record without sensor frames that carries the given prediction struct."""
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
    """Write the records into a table of the dataset schema and read them back."""
    frame = pl.DataFrame(
        [record.to_dictionary() for record in records],
        schema=DatasetTableSchema.to_polars_schema(),
    )
    return [DatasetRecord.load_from_dictionary(row) for row in frame.to_dicts()]


class TestPredictionSchema(unittest.TestCase):
    """The prediction struct schema matches its data model."""

    def test_struct_fields_match_the_data_model(self) -> None:
        """
        Input: the Polars field schema of the prediction struct and the fields of the model.
        Expected: the struct names the model fields in the same order, and the dictionary of a
        model holds exactly those names, so a record writes every field it declares.
        Check: the field name lists are equal and the dictionary keys equal the field names.
        """
        field_names = [field.name for field in PredictionDatasetSchema.to_polars_field_schema()]

        self.assertEqual(field_names, list(PredictionDataModel.model_fields))
        self.assertEqual(set(_prediction().to_dictionary()), set(field_names))


class TestPredictionRecordTable(unittest.TestCase):
    """The prediction column of the record table survives a write and a read."""

    def test_prediction_round_trips_through_the_record_table(self) -> None:
        """
        Input: one record carrying an Argoverse 2 prediction struct.
        Expected: the record read back from the table holds an equal prediction model with the
        format decoded to its enum, and the empty lidar frames stay empty.
        Check: the restored prediction, its dataset_format and the restored lidar frames.
        """
        prediction = _prediction()

        (restored,) = _round_trip([_record(prediction)])

        self.assertEqual(restored.prediction, prediction)
        self.assertIs(restored.prediction.dataset_format, DatasetFormat.AV2)
        self.assertEqual(restored.lidar_frames, [])

    def test_a_recording_record_round_trips(self) -> None:
        """
        Input: one record carrying a T4Dataset prediction struct, whose paths are None and whose
        window starts at step 12.
        Expected: the None paths and the window survive the table, so a recording dataset can
        describe a window over its frame records without naming files.
        Check: the restored prediction, its dataset_format, its two paths and its first step.
        """
        prediction = _recording_prediction()

        (restored,) = _round_trip([_record(prediction)])

        self.assertEqual(restored.prediction, prediction)
        self.assertIs(restored.prediction.dataset_format, DatasetFormat.T4)
        self.assertIsNone(restored.prediction.tracks_relative_path)
        self.assertIsNone(restored.prediction.map_relative_path)
        self.assertEqual(restored.prediction.first_step_index, 12)

    def test_a_perception_record_stores_a_null_struct(self) -> None:
        """
        Input: a record without a prediction next to a record with one, in the same table.
        Expected: the column is nullable, so the T4 / nuScenes generators never set it, and a
        record read back from the table materializes the null struct as an absent model while
        its neighbour keeps its prediction.
        Check: the dictionary value of the prediction column and the two restored predictions.
        """
        record = _record(None)

        self.assertIsNone(record.to_dictionary()[DatasetTableSchema.PREDICTION.name])
        restored, with_prediction = _round_trip([record, _record(_prediction())])
        self.assertIsNone(restored.prediction)
        self.assertEqual(with_prediction.prediction, _prediction())

    def test_the_record_defaults_to_no_prediction(self) -> None:
        """
        Input: a record built without the prediction argument.
        Expected: the field defaults to None, so the existing perception generators build their
        records unchanged.
        Check: the prediction attribute of the record.
        """
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

        self.assertIsNone(record.prediction)


class TestPredictionValidation(unittest.TestCase):
    """The prediction model rejects a source or a window it cannot serve."""

    def test_prediction_rejects_absolute_paths(self) -> None:
        """
        Input: an absolute tracks path, an absolute map path and an empty tracks path.
        Expected: a record never binds to one machine, so a path with a root is refused, as is
        an empty one.
        Check: ValueError naming the field and the reason.
        """
        with self.assertRaisesRegex(ValueError, "tracks_relative_path .* is absolute"):
            _prediction(tracks_relative_path="/data/av2/train/abc/scenario_abc.parquet")
        with self.assertRaisesRegex(ValueError, "map_relative_path .* is absolute"):
            _prediction(map_relative_path="/data/av2/train/abc/log_map_archive_abc.json")
        with self.assertRaisesRegex(ValueError, "tracks_relative_path is empty"):
            _prediction(tracks_relative_path="")

    def test_prediction_rejects_a_malformed_window(self) -> None:
        """
        Input: a negative first step, a current step at and below the window bounds, zero
        steps, a zero step period and a negative track count.
        Expected: every window field is range checked, so a window the loader could not slice
        is refused at record time.
        Check: ValueError naming the field and its bound.
        """
        with self.assertRaisesRegex(ValueError, "first_step_index must not be negative"):
            _prediction(first_step_index=-1)
        with self.assertRaisesRegex(ValueError, "current_step_index must lie in \\[0, 110\\)"):
            _prediction(current_step_index=110)
        with self.assertRaisesRegex(ValueError, "current_step_index must lie in"):
            _prediction(current_step_index=-1)
        with self.assertRaisesRegex(ValueError, "num_steps must be positive"):
            _prediction(num_steps=0, current_step_index=0)
        with self.assertRaisesRegex(ValueError, "step_seconds must be positive"):
            _prediction(step_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "num_tracks must not be negative"):
            _prediction(num_tracks=-1)

    def test_prediction_requires_the_source_fields_of_the_format(self) -> None:
        """
        Input: an Argoverse 2 record missing each of its paths and its focal id or carrying a
        scenario key, and a T4Dataset / nuScenes record carrying a path.
        Expected: a scenario dataset names its files and focal agent, a recording dataset names
        none, and a field the format has no use for is refused rather than ignored.
        Check: ValueError naming the field and the format.
        """
        with self.assertRaisesRegex(
            ValueError, "tracks_relative_path is required for the av2 format"
        ):
            _prediction(tracks_relative_path=None)
        with self.assertRaisesRegex(ValueError, "map_relative_path is required for the av2 format"):
            _prediction(map_relative_path=None)
        with self.assertRaisesRegex(ValueError, "focal_track_id is required for the av2 format"):
            _prediction(focal_track_id=None)
        with self.assertRaisesRegex(ValueError, "scenario_key must be None for the av2 format"):
            _prediction(scenario_key="0")
        with self.assertRaisesRegex(
            ValueError, "tracks_relative_path must be None for the t4 format"
        ):
            _recording_prediction(tracks_relative_path="scene/annotation")
        with self.assertRaisesRegex(ValueError, "map_relative_path must be None for the nuscenes"):
            _recording_prediction(
                dataset_format=DatasetFormat.NUSCENES, map_relative_path="map.json"
            )

    def test_prediction_is_strict_about_the_format_enum(self) -> None:
        """
        Input: the model built with the string "av2" and a dictionary naming the unknown format
        "womd".
        Expected: the model itself takes the enum and only load_from_dictionary decodes the
        stored string, so a stray string or an unknown format fails loudly.
        Check: ValueError from the constructor and from load_from_dictionary.
        """
        with self.assertRaises(ValueError):
            _prediction(dataset_format="av2")
        with self.assertRaises(ValueError):
            PredictionDataModel.load_from_dictionary(
                _prediction().to_dictionary() | {"dataset_format": "womd"}
            )
