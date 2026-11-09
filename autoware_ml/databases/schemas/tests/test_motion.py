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

"""Tests for the motion schema and its column of the record table."""

from __future__ import annotations

from typing import Any
import unittest

import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.schemas.motion import MotionDataModel, MotionDatasetSchema
from autoware_ml.types.dataset import DatasetFormat


def _motion(**overrides: Any) -> MotionDataModel:
    """Build an Argoverse 2 motion record with the given fields overridden."""
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
    return MotionDataModel(**(fields | overrides))


def _recording_motion(**overrides: Any) -> MotionDataModel:
    """Build a T4Dataset motion record with the given fields overridden.

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
    return MotionDataModel(**(fields | overrides))


def _record(motion: MotionDataModel | None) -> DatasetRecord:
    """Build a record without sensor frames that carries the given motion struct."""
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
        motion=motion,
    )


def _round_trip(records: list[DatasetRecord]) -> list[DatasetRecord]:
    """Write the records into a table of the dataset schema and read them back."""
    frame = pl.DataFrame(
        [record.to_dictionary() for record in records],
        schema=DatasetTableSchema.to_polars_schema(),
    )
    return [DatasetRecord.load_from_dictionary(row) for row in frame.to_dicts()]


class TestMotionSchema(unittest.TestCase):
    """The motion struct schema matches its data model."""

    def test_struct_fields_match_the_data_model(self) -> None:
        """
        Input: the Polars field schema of the motion struct and the fields of the model.
        Expected: the struct names the model fields in the same order, and the dictionary of a
        model holds exactly those names, so a record writes every field it declares.
        Check: the field name lists are equal and the dictionary keys equal the field names.
        """
        field_names = [field.name for field in MotionDatasetSchema.to_polars_field_schema()]

        self.assertEqual(field_names, list(MotionDataModel.model_fields))
        self.assertEqual(set(_motion().to_dictionary()), set(field_names))


class TestMotionRecordTable(unittest.TestCase):
    """The motion column of the record table survives a write and a read."""

    def test_motion_round_trips_through_the_record_table(self) -> None:
        """
        Input: one record carrying an Argoverse 2 motion struct.
        Expected: the record read back from the table holds an equal motion model with the
        format decoded to its enum, and the empty lidar frames stay empty.
        Check: the restored motion, its dataset_format and the restored lidar frames.
        """
        motion = _motion()

        (restored,) = _round_trip([_record(motion)])

        self.assertEqual(restored.motion, motion)
        self.assertIs(restored.motion.dataset_format, DatasetFormat.AV2)
        self.assertEqual(restored.lidar_frames, [])

    def test_a_recording_record_round_trips(self) -> None:
        """
        Input: one record carrying a T4Dataset motion struct, whose paths are None and whose
        window starts at step 12.
        Expected: the None paths and the window survive the table, so a recording dataset can
        describe a window over its frame records without naming files.
        Check: the restored motion, its dataset_format, its two paths and its first step.
        """
        motion = _recording_motion()

        (restored,) = _round_trip([_record(motion)])

        self.assertEqual(restored.motion, motion)
        self.assertIs(restored.motion.dataset_format, DatasetFormat.T4)
        self.assertIsNone(restored.motion.tracks_relative_path)
        self.assertIsNone(restored.motion.map_relative_path)
        self.assertEqual(restored.motion.first_step_index, 12)

    def test_a_perception_record_stores_a_null_struct(self) -> None:
        """
        Input: a record without a motion next to a record with one, in the same table.
        Expected: the column is nullable, so the T4 / nuScenes generators never set it, and a
        record read back from the table materializes the null struct as an absent model while
        its neighbour keeps its motion.
        Check: the dictionary value of the motion column and the two restored motions.
        """
        record = _record(None)

        self.assertIsNone(record.to_dictionary()[DatasetTableSchema.MOTION.name])
        restored, with_motion = _round_trip([record, _record(_motion())])
        self.assertIsNone(restored.motion)
        self.assertEqual(with_motion.motion, _motion())

    def test_the_record_defaults_to_no_motion(self) -> None:
        """
        Input: a record built without the motion argument.
        Expected: the field defaults to None, so the existing perception generators build their
        records unchanged.
        Check: the motion attribute of the record.
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

        self.assertIsNone(record.motion)


class TestMotionValidation(unittest.TestCase):
    """The motion model rejects a source or a window it cannot serve."""

    def test_motion_rejects_absolute_paths(self) -> None:
        """
        Input: an absolute tracks path, an absolute map path and an empty tracks path.
        Expected: a record never binds to one machine, so a path with a root is refused, as is
        an empty one.
        Check: ValueError naming the field and the reason.
        """
        with self.assertRaisesRegex(ValueError, "tracks_relative_path .* is absolute"):
            _motion(tracks_relative_path="/data/av2/train/abc/scenario_abc.parquet")
        with self.assertRaisesRegex(ValueError, "map_relative_path .* is absolute"):
            _motion(map_relative_path="/data/av2/train/abc/log_map_archive_abc.json")
        with self.assertRaisesRegex(ValueError, "tracks_relative_path is empty"):
            _motion(tracks_relative_path="")

    def test_motion_rejects_a_malformed_window(self) -> None:
        """
        Input: a negative first step, a current step at and below the window bounds, zero
        steps, a zero step period and a negative track count.
        Expected: every window field is range checked, so a window the loader could not slice
        is refused at record time.
        Check: ValueError naming the field and its bound.
        """
        with self.assertRaisesRegex(ValueError, "first_step_index must not be negative"):
            _motion(first_step_index=-1)
        with self.assertRaisesRegex(ValueError, "current_step_index must lie in \\[0, 110\\)"):
            _motion(current_step_index=110)
        with self.assertRaisesRegex(ValueError, "current_step_index must lie in"):
            _motion(current_step_index=-1)
        with self.assertRaisesRegex(ValueError, "num_steps must be positive"):
            _motion(num_steps=0, current_step_index=0)
        with self.assertRaisesRegex(ValueError, "step_seconds must be positive"):
            _motion(step_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "num_tracks must not be negative"):
            _motion(num_tracks=-1)

    def test_motion_requires_the_source_fields_of_the_format(self) -> None:
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
            _motion(tracks_relative_path=None)
        with self.assertRaisesRegex(ValueError, "map_relative_path is required for the av2 format"):
            _motion(map_relative_path=None)
        with self.assertRaisesRegex(ValueError, "focal_track_id is required for the av2 format"):
            _motion(focal_track_id=None)
        with self.assertRaisesRegex(ValueError, "scenario_key must be None for the av2 format"):
            _motion(scenario_key="0")
        with self.assertRaisesRegex(
            ValueError, "tracks_relative_path must be None for the t4 format"
        ):
            _recording_motion(tracks_relative_path="scene/annotation")
        with self.assertRaisesRegex(ValueError, "map_relative_path must be None for the nuscenes"):
            _recording_motion(dataset_format=DatasetFormat.NUSCENES, map_relative_path="map.json")

    def test_motion_is_strict_about_the_format_enum(self) -> None:
        """
        Input: the model built with the string "av2" and a dictionary naming the unknown format
        "womd".
        Expected: the model itself takes the enum and only load_from_dictionary decodes the
        stored string, so a stray string or an unknown format fails loudly.
        Check: ValueError from the constructor and from load_from_dictionary.
        """
        with self.assertRaises(ValueError):
            _motion(dataset_format="av2")
        with self.assertRaises(ValueError):
            MotionDataModel.load_from_dictionary(
                _motion().to_dictionary() | {"dataset_format": "womd"}
            )
