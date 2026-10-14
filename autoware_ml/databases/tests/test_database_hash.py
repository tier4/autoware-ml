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

import yaml

from autoware_ml.databases.scenarios import DatasetParams
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
    "train": ["db_scenario_a/0/tokyo/j6gen2/none", "db_scenario_b/1/tokyo/j6gen2/none"],
    "val": ["db_scenario_c/0/tokyo/j6gen2/none"],
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


class TestDatabaseHash(unittest.TestCase):
    """Unit tests for the content hash that keys the database cache file."""

    def setUp(self) -> None:
        """Write the scenario yaml the database is built from into a temporary directory."""
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.scenario_root_path = Path(self._tmp_dir.name) / "scenarios"
        self.scenario_root_path.mkdir()
        with open(self.scenario_root_path / "db_test.yaml", "w") as f:
            yaml.safe_dump(_SCENARIOS, f)

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
        the dataset, and the cache happen to live. Cached records are re-rooted against the
        database root path when they are read back, so a path in the hash would hand every
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

        smaller_scenarios = dict(_SCENARIOS, train=_SCENARIOS["train"][:1])
        with open(self.scenario_root_path / "db_test.yaml", "w") as f:
            yaml.safe_dump(smaller_scenarios, f)

        self.assertNotEqual(reference, self._build_database().database_hash)

    def test_hash_ignores_debug_representation(self) -> None:
        """
        Test that the hash does not depend on ``__str__``, which is a debug representation that
        carries paths and is free to gain fields without invalidating every cached file.
        """
        database = self._build_database()
        self.assertIn(str(database.version), database.hash_repr)
        self.assertNotIn(str(self.scenario_root_path), database.hash_repr)
        self.assertNotIn(str(database._root_path), database.hash_repr)
        self.assertNotIn(str(database._cache_path), database.hash_repr)


if __name__ == "__main__":
    unittest.main()
