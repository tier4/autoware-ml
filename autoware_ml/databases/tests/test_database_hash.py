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

"""Unit tests for the database cache hash."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from autoware_ml.databases.nuscenes import nuscenes_dataset
from autoware_ml.databases.nuscenes.nuscenes_dataset import NuScenesDataset
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.t4dataset import t4dataset
from autoware_ml.databases.t4dataset.t4dataset import T4Dataset
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.taxonomy import (
    DatabaseTaxonomy,
    DetectionTaxonomy,
    LabelVocabulary,
    SegmentationTaxonomy,
)
from autoware_ml.types.metrics import AgentKind

_SCENARIOS = {
    "train": ["db_scenario_a/0/tokyo/j6gen2/true", "db_scenario_b/1/tokyo/j6gen2/true"],
    "val": ["db_scenario_c/0/tokyo/j6gen2/true"],
}


def _build_taxonomy(class_names: tuple[str, ...]) -> DatabaseTaxonomy:
    """Build a taxonomy over the class names, with buses folded into cars."""
    vocabulary = LabelVocabulary({name: name for name in class_names + ("bus",)})
    class_mapping = {name: name for name in class_names} | {"bus": "car"}
    class_groups = {"all": list(class_names)}
    return DatabaseTaxonomy(
        detection3d=DetectionTaxonomy(
            vocabulary=vocabulary,
            class_names=class_names,
            class_mapping=class_mapping,
            ignore_index=-1,
            class_groups=class_groups,
            eval_range={name: 50.0 for name in class_names},
            collision_kinds={name: AgentKind.WHEELED for name in class_names},
            living_speeds={},
        ),
        segmentation3d=SegmentationTaxonomy(
            vocabulary=vocabulary,
            class_names=class_names,
            class_mapping=class_mapping,
            ignore_index=-1,
            class_groups=class_groups,
        ),
    )


class TestScenarioDataHashRepr(unittest.TestCase):
    """The content representation of a scenario lists every field the records depend on."""

    def _scenario(self, status: bool | None) -> ScenarioData:
        return ScenarioData(
            dataset_params=DatasetParams(dataset_name="db_test", sample_steps=2),
            scenario_id="scene_a",
            scenario_version="0",
            vehicle_type="j6gen2",
            location="tokyo",
            traffic_cone_barrier_bbox_status=status,
        )

    def test_lists_every_field_explicitly(self) -> None:
        """Test that the scenario id, version, vehicle, location, sampling step and the traffic
        cone / barrier flag all appear by name, so none can go missing silently."""
        hash_repr = self._scenario(True).hash_repr

        for field in (
            "dataset_name='db_test'",
            "sample_steps=2",
            "scenario_id='scene_a'",
            "scenario_version='0'",
            "vehicle_type='j6gen2'",
            "location='tokyo'",
            "traffic_cone_barrier_bbox_status=True",
        ):
            with self.subTest(field=field):
                self.assertIn(field, hash_repr)

    def test_the_flag_tells_scenarios_apart(self) -> None:
        """Test that the flag, which the records carry per frame, changes the representation."""
        self.assertNotEqual(self._scenario(True).hash_repr, self._scenario(False).hash_repr)
        self.assertNotEqual(self._scenario(True).hash_repr, self._scenario(None).hash_repr)


class TestT4DatabaseHash(unittest.TestCase):
    """Unit tests for the content hash that keys the T4 database cache file."""

    def setUp(self) -> None:
        """Write the scenario yaml the database is built from into a temporary directory."""
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.scenario_root_path = Path(self._tmp_dir.name) / "scenarios"
        self.scenario_root_path.mkdir()
        self._write_scenarios(_SCENARIOS)

    def _write_scenarios(self, scenarios: dict[str, list[str]]) -> None:
        with open(self.scenario_root_path / "db_test.yaml", "w") as f:
            yaml.safe_dump(scenarios, f)

    def _build_database(
        self,
        scenario_root_path: Path | None = None,
        root_path: str = "/data/t4dataset",
        cache_path: str | None = None,
        version: str = "T4Dataset-test-v1.0.0",
        class_names: tuple[str, ...] = ("car", "pedestrian"),
        lidar_channel: str = "LIDAR_CONCAT",
        lidar_pointcloud_num_features: int = 5,
        box_annotation_dir: str = "annotation",
    ) -> T4Dataset:
        """Build a T4Dataset whose location and content settings can be varied independently."""
        scenarios = T4Scenarios(
            scenario_root_path=scenario_root_path or self.scenario_root_path,
            dataset_params=[
                DatasetParams(dataset_name="db_test", sample_steps=1),
            ],
        )
        return T4Dataset(
            version=version,
            root_path=root_path,
            scenarios={"db_test": scenarios},
            cache_path=cache_path or str(Path(self._tmp_dir.name) / "cache"),
            cache_file_prefix_name="database",
            num_workers=1,
            taxonomy=_build_taxonomy(class_names),
            lidar_channel=lidar_channel,
            lidar_pointcloud_num_features=lidar_pointcloud_num_features,
            box3d_pipelines=[],
            lidar_intensity_scale=255.0,
            box_annotation_dir=box_annotation_dir,
        )

    def test_hash_ignores_filesystem_locations(self) -> None:
        """
        Test that the same database content hashes the same regardless of where the annotations,
        the dataset, and the cache happen to live. T4 records store paths relative to the
        database root and are re-rooted when read back, so a path in the hash would hand every
        machine its own cache file for identical content.
        """
        # The same scenario yaml, reachable under a second path
        alternative_root = Path(self._tmp_dir.name) / "elsewhere"
        alternative_root.mkdir()
        (alternative_root / "db_test.yaml").write_bytes(
            (self.scenario_root_path / "db_test.yaml").read_bytes()
        )

        reference = self._build_database().database_hash
        relocated = self._build_database(
            scenario_root_path=alternative_root,
            root_path="/mnt/somewhere/else/t4dataset",
            cache_path=str(Path(self._tmp_dir.name) / "other_cache"),
        ).database_hash

        self.assertEqual(reference, relocated)

    def test_hash_tracks_content_settings(self) -> None:
        """
        Test that every setting that changes what lands in the cache changes the hash, so a
        stale cache is never mistaken for a matching one.
        """
        reference = self._build_database().database_hash

        self.assertNotEqual(
            reference, self._build_database(version="T4Dataset-test-v2.0.0").database_hash
        )
        self.assertNotEqual(reference, self._build_database(class_names=("car",)).database_hash)
        self.assertNotEqual(
            reference, self._build_database(lidar_channel="LIDAR_TOP").database_hash
        )
        self.assertNotEqual(
            reference, self._build_database(lidar_pointcloud_num_features=4).database_hash
        )
        self.assertNotEqual(
            reference, self._build_database(box_annotation_dir="annotation_v2").database_hash
        )

    def test_hash_tracks_scenario_selection(self) -> None:
        """Test that selecting a different set of scenarios yields a different hash."""
        reference = self._build_database().database_hash

        self._write_scenarios(dict(_SCENARIOS, train=_SCENARIOS["train"][:1]))

        self.assertNotEqual(reference, self._build_database().database_hash)

    def test_hash_tracks_the_traffic_cone_barrier_flag(self) -> None:
        """
        Test that the same scenarios declared with another traffic cone / barrier annotation
        status hash differently, since the records carry the flag on every frame.
        """
        reference = self._build_database().database_hash

        self._write_scenarios(
            {
                split: [path.replace("/true", "/false") for path in paths]
                for split, paths in _SCENARIOS.items()
            }
        )

        self.assertNotEqual(reference, self._build_database().database_hash)

    def test_hash_tracks_the_generator_sweep_window(self) -> None:
        """
        Test that the sweep window the records generator stores per sample is part of the hash,
        so a changed constant does not reuse records built with the old window.
        """
        reference = self._build_database().database_hash

        with patch.object(t4dataset, "MAX_SWEEPS", 3):
            narrowed = self._build_database().database_hash

        self.assertNotEqual(reference, narrowed)

    def test_hash_is_built_from_explicit_fields_not_the_debug_representation(self) -> None:
        """
        Test that the hash lists its content fields by name and does not depend on ``__str__``,
        which is a debug representation that carries paths and is free to gain fields.
        """
        database = self._build_database()

        fields = database.hash_fields
        self.assertEqual(
            list(fields),
            [
                "database",
                "version",
                "taxonomy",
                "box3d_pipelines",
                "lidar_pointcloud_num_features",
                "scenarios",
                "lidar_channel",
                "box_annotation_dir",
                "max_sweeps",
            ],
        )
        self.assertEqual(fields["version"], database.version)
        self.assertEqual(fields["lidar_pointcloud_num_features"], "5")
        self.assertIn("traffic_cone_barrier_bbox_status=True", fields["scenarios"])
        self.assertNotIn(str(self.scenario_root_path), database.hash_repr)
        self.assertNotIn(str(database._root_path), database.hash_repr)
        self.assertNotIn(str(database._cache_path), database.hash_repr)


class TestNuScenesDatabaseHash(unittest.TestCase):
    """Unit tests for the content hash that keys the nuScenes database cache file."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)

    def _build_database(
        self,
        root_path: str = "/data/nuscenes",
        cache_path: str | None = None,
        lidar_pointcloud_num_features: int = 5,
    ) -> NuScenesDataset:
        """Build a NuScenesDataset without scenarios, which need the devkit tables."""
        return NuScenesDataset(
            version="v1.0-mini",
            root_path=root_path,
            scenarios={},
            cache_path=cache_path or str(Path(self._tmp_dir.name) / "cache"),
            cache_file_prefix_name="database",
            num_workers=1,
            taxonomy=_build_taxonomy(("car", "pedestrian")),
            lidar_pointcloud_num_features=lidar_pointcloud_num_features,
            box3d_pipelines=[],
            lidar_intensity_scale=255.0,
        )

    def test_hash_tracks_the_feature_count(self) -> None:
        """
        Test that the number of stored point features is part of the hash for every database,
        not only T4: the records are read by that count.
        """
        self.assertNotEqual(
            self._build_database().database_hash,
            self._build_database(lidar_pointcloud_num_features=4).database_hash,
        )

    def test_hash_ignores_filesystem_locations(self) -> None:
        """
        Test that neither the root path nor the cache path is part of the nuScenes hash: the
        records store every file path relative to the root and the datamodule joins them onto
        the root of the node reading them, so the same records serve any root.
        """
        reference = self._build_database().database_hash

        self.assertEqual(reference, self._build_database(root_path="/mnt/nuscenes").database_hash)
        self.assertEqual(
            reference,
            self._build_database(cache_path=str(Path(self._tmp_dir.name) / "other")).database_hash,
        )
        self.assertNotIn("/data/nuscenes", self._build_database().hash_repr)

    def test_hash_tracks_the_generator_sweep_window(self) -> None:
        """Test that the sweep window the nuScenes generator stores per sample is in the hash."""
        reference = self._build_database().database_hash

        with patch.object(nuscenes_dataset, "MAX_SWEEPS", 3):
            narrowed = self._build_database().database_hash

        self.assertNotEqual(reference, narrowed)


if __name__ == "__main__":
    unittest.main()
