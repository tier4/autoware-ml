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

"""Unit tests for the nuScenes records generator, built over a one-scene devkit stub."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

from autoware_ml.databases.nuscenes import nuscenes_records_generator
from autoware_ml.databases.nuscenes.nuscenes_records_generator import NuScenesRecordsGenerator
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.schemas.frame_basic_metadata import FrameBasicMetadata
from autoware_ml.databases.taxonomy import DatabaseTaxonomy

_DATASET_NAME = "nuscenes"
_VERSION = "v1.0-mini"
_SCENE_NAME = "scene-0061"
_SCENE_TOKEN = "scene_token"
_FIRST_SAMPLE_TOKEN = "sample0"
_SECOND_SAMPLE_TOKEN = "sample1"
_DATAROOT = "/data/nuscenes"


class _SceneDevkit:
    """Devkit stub with one scene of two chained samples, as the generator walks them."""

    def __init__(self) -> None:
        self.scene = [
            {"token": _SCENE_TOKEN, "name": _SCENE_NAME, "first_sample_token": _FIRST_SAMPLE_TOKEN}
        ]
        self._samples = {
            _FIRST_SAMPLE_TOKEN: {
                "token": _FIRST_SAMPLE_TOKEN,
                "timestamp": 1_500_000,
                "next": _SECOND_SAMPLE_TOKEN,
            },
            _SECOND_SAMPLE_TOKEN: {
                "token": _SECOND_SAMPLE_TOKEN,
                "timestamp": 2_000_000,
                "next": "",
            },
        }

        self.dataroot = _DATAROOT
        self.table_names = ["sample", "lidarseg"]
        self._lidarseg = {
            _FIRST_SAMPLE_TOKEN: {"filename": "lidarseg/v1.0-mini/first_lidar_token.bin"}
        }

    def get(self, table_name: str, token: str) -> Mapping[str, Any]:
        if table_name == "sample":
            return self._samples[token]
        if table_name == "lidarseg":
            return self._lidarseg[token]
        raise KeyError(table_name)

    def get_sample_data_path(self, token: str) -> str:
        """Resolve a file under the dataroot, as the devkit does."""
        return f"{self.dataroot}/samples/LIDAR_TOP/{token}.pcd.bin"


class TestNuScenesRecordsGenerator(unittest.TestCase):
    """The generator builds complete frame metadata for every nuScenes sample."""

    def setUp(self) -> None:
        """Replace the devkit with the one-scene stub and build the generator over it."""
        self.devkit = _SceneDevkit()
        devkit_patch = patch.object(
            nuscenes_records_generator,
            "NuScenes",
            lambda version, dataroot, verbose: self.devkit,
        )
        devkit_patch.start()
        self.addCleanup(devkit_patch.stop)
        self.generator = NuScenesRecordsGenerator(
            database_root_path="/data/nuscenes",
            scenario_data=ScenarioData(
                dataset_params=DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1),
                scenario_id=_SCENE_NAME,
                scenario_version=_VERSION,
            ),
            lidar_pointcloud_num_features=5,
            taxonomy=MagicMock(spec=DatabaseTaxonomy),
            box3d_pipelines=[],
        )

    def test_walks_the_samples_of_the_scene_in_order(self) -> None:
        """Test that the scene record is found by name and its samples are chained by `next`."""
        self.assertEqual(self.generator.scene_record["name"], _SCENE_NAME)
        self.assertEqual(
            list(self.generator.sample_tokens), [_FIRST_SAMPLE_TOKEN, _SECOND_SAMPLE_TOKEN]
        )

    def test_basic_metadata_is_complete_and_flags_cones_and_barriers_as_annotated(self) -> None:
        """
        Test that the basic metadata validates under the strict, required-field schema, with the
        traffic cone / barrier flag set since nuScenes annotates both in every scene. The field
        has no default, so leaving it out would raise a ValidationError on every sample.
        """
        sample = self.devkit.get("sample", _SECOND_SAMPLE_TOKEN)

        metadata = self.generator._extract_sample_basic_metadata(sample, sample_index=1)

        self.assertIsInstance(metadata, FrameBasicMetadata)
        self.assertEqual(
            metadata,
            FrameBasicMetadata(
                scenario_id=_SCENE_NAME,
                sample_id=_SECOND_SAMPLE_TOKEN,
                sample_index=1,
                timestamp_seconds=2.0,
                scenario_name=_SCENE_NAME,
                location=None,
                vehicle_type=None,
                traffic_cone_barrier_bbox_status=True,
            ),
        )

    def test_stored_paths_are_relative_to_the_database_root(self) -> None:
        """
        Test that the paths the devkit resolves under the dataroot are stored relative to it,
        so the records stay valid when the database is mounted under another root and the
        datamodule joins them back onto its own root.
        """
        resolved = self.devkit.get_sample_data_path(_FIRST_SAMPLE_TOKEN)

        self.assertEqual(
            self.generator.relative_to_database_root(resolved),
            f"samples/LIDAR_TOP/{_FIRST_SAMPLE_TOKEN}.pcd.bin",
        )
        self.assertEqual(
            self.generator._extract_lidar_pointcloud_semantic_mask_path(_FIRST_SAMPLE_TOKEN),
            "lidarseg/v1.0-mini/first_lidar_token.bin",
        )

    def test_a_path_outside_the_database_root_is_rejected(self) -> None:
        """Test that a file the devkit resolves elsewhere cannot be stored relative to the root."""
        with self.assertRaisesRegex(ValueError, "does not lie under the database root"):
            self.generator.relative_to_database_root("/elsewhere/samples/LIDAR_TOP/x.pcd.bin")

    def test_unknown_scene_is_rejected(self) -> None:
        """Test that a scenario naming a scene the version does not hold is an error."""
        with self.assertRaisesRegex(ValueError, "not found"):
            NuScenesRecordsGenerator(
                database_root_path="/data/nuscenes",
                scenario_data=ScenarioData(
                    dataset_params=DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1),
                    scenario_id="scene-9999",
                    scenario_version=_VERSION,
                ),
                lidar_pointcloud_num_features=5,
                taxonomy=MagicMock(spec=DatabaseTaxonomy),
                box3d_pipelines=[],
            )


if __name__ == "__main__":
    unittest.main()
