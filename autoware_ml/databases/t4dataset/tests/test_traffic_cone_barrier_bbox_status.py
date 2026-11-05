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

"""Unit tests following the traffic cone / barrier annotation flag from the scenario data
through the frame metadata, the sample record, and the dataset record table."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from t4_devkit.schema import SchemaName

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.schemas.frame_basic_metadata import FrameBasicMetadata
from autoware_ml.databases.t4dataset import t4records_generator
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.databases.t4dataset.t4sample_records import T4SampleRecord
from autoware_ml.databases.taxonomy import DatabaseTaxonomy
from autoware_ml.types.sensor import LidarChannel

_DATASET_NAME = "db_test_v1"
_SCENARIO_ID = "scene_a"
_SCENARIO_VERSION = "0"
_SCENE_TOKEN = "scene_token"
_SCENE_NAME = "scene_a_name"
_SAMPLE_TOKEN = "sample0"
_COLUMN = DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name


@dataclass(frozen=True)
class _Sample:
    """Sample record with the fields the basic metadata extraction reads."""

    token: str
    timestamp: int
    scene_token: str


@dataclass(frozen=True)
class _Scene:
    """Scene record with the fields the basic metadata extraction reads."""

    token: str
    name: str


class _SceneDevkit:
    """Devkit stub holding one scene and the empty tables the generator indexes at construction."""

    def __init__(self) -> None:
        self.lidarseg = []
        self.sample = [_Sample(token=_SAMPLE_TOKEN, timestamp=2_000_000, scene_token=_SCENE_TOKEN)]
        self._scenes = {_SCENE_TOKEN: _Scene(token=_SCENE_TOKEN, name=_SCENE_NAME)}

    def get(self, schema: SchemaName, token: str) -> _Scene:
        if schema != SchemaName.SCENE:
            raise KeyError(schema)
        return self._scenes[token]


def _frame_metadata(status: bool | None) -> FrameBasicMetadata:
    """Build frame metadata carrying ``status``."""
    return FrameBasicMetadata(
        scenario_id=_SCENARIO_ID,
        sample_id=_SAMPLE_TOKEN,
        sample_index=0,
        timestamp_seconds=2.0,
        scenario_name=_SCENE_NAME,
        location="city_a",
        vehicle_type="vehicle_a",
        traffic_cone_barrier_bbox_status=status,
    )


def _dataset_record(status: bool | None) -> DatasetRecord:
    """Build a dataset record without frames or boxes carrying ``status``."""
    return DatasetRecord(
        scenario_id=_SCENARIO_ID,
        sample_id=_SAMPLE_TOKEN,
        sample_index=0,
        timestamp_seconds=2.0,
        location="city_a",
        vehicle_type="vehicle_a",
        scenario_name=_SCENE_NAME,
        traffic_cone_barrier_bbox_status=status,
        lidar_frames=[],
        lidar_sources=None,
        image_frames=None,
        category_mapping=None,
        boxes_3d=None,
    )


class TestRecordsGenerator(unittest.TestCase):
    """The generator copies the scenario flag into the basic metadata of every frame."""

    def setUp(self) -> None:
        """Lay out the scene directory and replace the devkit with the one-scene stub."""
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.database_root_path = Path(self._tmp_dir.name)
        scene_dir = self.database_root_path / _DATASET_NAME / _SCENARIO_ID / _SCENARIO_VERSION
        (scene_dir / "annotation").mkdir(parents=True)
        self.devkit = _SceneDevkit()
        devkit_patch = patch.object(
            t4records_generator, "T4Devkit", lambda data_root, verbose: self.devkit
        )
        table_patch = patch.object(t4records_generator, "load_table", lambda path, schema: [])
        devkit_patch.start()
        table_patch.start()
        self.addCleanup(devkit_patch.stop)
        self.addCleanup(table_patch.stop)

    def _build_generator(self, status: bool | None) -> T4RecordsGenerator:
        """Build a generator over a scenario declaring ``status``."""
        return T4RecordsGenerator(
            database_root_path=str(self.database_root_path),
            scenario_data=ScenarioData(
                dataset_params=DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1),
                scenario_id=_SCENARIO_ID,
                scenario_version=_SCENARIO_VERSION,
                vehicle_type="vehicle_a",
                location="city_a",
                traffic_cone_barrier_bbox_status=status,
            ),
            lidar_channel=LidarChannel.LIDAR_TOP,
            lidar_pointcloud_num_features=5,
            box_annotation_dir="annotation",
            taxonomy=MagicMock(spec=DatabaseTaxonomy),
            box3d_pipelines=[],
        )

    def test_scenario_flag_lands_in_the_frame_metadata(self) -> None:
        """Annotated, not annotated, and unknown scenarios each stamp their frames accordingly."""
        for status in (True, False, None):
            with self.subTest(status=status):
                generator = self._build_generator(status)

                metadata = generator._extract_sample_basic_metadata(self.devkit.sample[0], 0)

                self.assertIs(metadata.traffic_cone_barrier_bbox_status, status)
                self.assertEqual(metadata.scenario_id, _SCENARIO_ID)
                self.assertEqual(metadata.scenario_name, _SCENE_NAME)
                self.assertEqual(metadata.location, "city_a")
                self.assertEqual(metadata.vehicle_type, "vehicle_a")
                self.assertAlmostEqual(metadata.timestamp_seconds, 2.0)


class TestSampleRecord(unittest.TestCase):
    """The sample record forwards the frame flag into the dataset record."""

    def test_flag_is_copied_into_the_dataset_record(self) -> None:
        """Each flag value reaches the dataset record unchanged."""
        for status in (True, False, None):
            with self.subTest(status=status):
                sample_record = T4SampleRecord(
                    frame_basic_metadata=_frame_metadata(status),
                    lidar_frame_data_models=[],
                    lidar_source_data_models=[],
                    image_frame_data_models=[],
                    category_mapping_data_model=CategoryMappingDataModel(
                        category_names=["car"], category_indices=[0]
                    ),
                    boxes_3d_data_model=[],
                )

                record = sample_record.to_dataset_record()

                self.assertIs(record.traffic_cone_barrier_bbox_status, status)


class TestDatasetRecordTable(unittest.TestCase):
    """The record table declares the flag as a nullable boolean column and round-trips it."""

    def test_schema_declares_a_boolean_column(self) -> None:
        """The flag is a boolean column of the record table."""
        schema = DatasetTableSchema.to_polars_schema()

        self.assertEqual(schema[_COLUMN], pl.Boolean)

    def test_flag_defaults_to_unknown(self) -> None:
        """A record built without the flag reports it as unknown rather than as a value."""
        record = DatasetRecord(
            scenario_id=_SCENARIO_ID,
            sample_id=_SAMPLE_TOKEN,
            sample_index=0,
            timestamp_seconds=2.0,
            location=None,
            vehicle_type=None,
            scenario_name=_SCENE_NAME,
            lidar_frames=[],
            lidar_sources=None,
            image_frames=None,
            category_mapping=None,
            boxes_3d=None,
        )

        self.assertIsNone(record.traffic_cone_barrier_bbox_status)

    def test_flag_round_trips_through_the_table_dictionary(self) -> None:
        """Serialising and loading a record preserves every flag value, including unknown."""
        for status in (True, False, None):
            with self.subTest(status=status):
                dictionary = _dataset_record(status).to_dictionary()

                self.assertIs(dictionary[_COLUMN], status)
                self.assertIs(
                    DatasetRecord.load_from_dictionary(dictionary).traffic_cone_barrier_bbox_status,
                    status,
                )

    def test_flag_survives_the_polars_table(self) -> None:
        """Rows written with the table schema read the flag back, nulls included."""
        rows = [_dataset_record(status).to_dictionary() for status in (True, False, None)]

        table = pl.DataFrame(rows, schema=DatasetTableSchema.to_polars_schema())

        self.assertEqual(table[_COLUMN].to_list(), [True, False, None])
        loaded = DatasetRecord.load_from_dictionary(table.row(1, named=True))
        self.assertIs(loaded.traffic_cone_barrier_bbox_status, False)


if __name__ == "__main__":
    unittest.main()
