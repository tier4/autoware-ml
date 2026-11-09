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

"""Tests for the Argoverse 2 motion forecasting database: its cache hash and its records."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoware_ml.databases.argoverse2 import av2_motion_database, av2_motion_records_generator
from autoware_ml.databases.argoverse2.av2_motion_database import AV2MotionDatabase
from autoware_ml.databases.argoverse2.av2_motion_scenarios import AV2MotionScenarios
from autoware_ml.databases.argoverse2.tests.av2_fixtures import av2_vocabulary, write_scenario
from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.taxonomy import LabelVocabulary
from autoware_ml.types.dataset import SplitType

_TRAIN_IDS = [f"{index:04d}" for index in range(6)]
_VAL_IDS = ["v000", "v001"]

# The hash tests work on empty scenario directories; only the records test reads real files
_AV2_AVAILABLE = importlib.util.find_spec("av2") is not None


def _write_empty_corpus(root: Path, splits: dict[str, list[str]]) -> Path:
    """Scenario directories without files, enough to discover scenarios and hash them."""
    for split, scenario_ids in splits.items():
        for scenario_id in scenario_ids:
            (root / split / scenario_id).mkdir(parents=True)
    return root


def _database(
    root: Path,
    cache_path: Path,
    version: str = "AV2MotionForecasting-test-v1.0.0",
    sample_steps: int = 1,
    splits: tuple[str, ...] = ("train", "val"),
    vocabulary: LabelVocabulary | None = None,
    num_workers: int = 1,
    scenarios_per_worker_task: int = 256,
) -> AV2MotionDatabase:
    """Build the database of the corpus with the given settings overridden."""
    scenarios = AV2MotionScenarios(
        scenario_root_path=root,
        dataset_params=[
            DatasetParams(dataset_name=split, sample_steps=sample_steps) for split in splits
        ],
    )
    return AV2MotionDatabase(
        version=version,
        root_path=str(root),
        scenarios={"av2": scenarios},
        cache_path=str(cache_path),
        cache_file_prefix_name="database",
        num_workers=num_workers,
        vocabulary=vocabulary or av2_vocabulary(),
        scenarios_per_worker_task=scenarios_per_worker_task,
    )


class AV2MotionDatabaseTestCase(unittest.TestCase):
    """Base case with a temporary directory, an empty corpus and a cache path per test."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name)
        self.corpus = _write_empty_corpus(
            self.tmp_path / "av2", {"train": _TRAIN_IDS, "val": _VAL_IDS}
        )
        self.cache_path = self.tmp_path / "cache"


class TestAV2MotionDatabaseHash(AV2MotionDatabaseTestCase):
    """The cache hash keys the content of the records and nothing else."""

    def test_rejects_an_empty_worker_task(self) -> None:
        """
        Input: the database built with zero scenarios per worker task.
        Expected: a worker task holds at least one scenario.
        Check: ValueError naming the setting.
        """
        with self.assertRaisesRegex(ValueError, "scenarios_per_worker_task must be positive"):
            _database(self.corpus, self.cache_path, scenarios_per_worker_task=0)

    def test_hash_fields_and_compact_representations(self) -> None:
        """
        Input: the database of the empty corpus with six train and two val scenarios.
        Expected: the hash fields list the base fields in order, with the absent box
        pipelines and lidar settings rendered as None, followed by the observed steps and the
        dataset format; neither representation lists the scenario ids; two databases of the
        same corpus are equal.
        Check: the field names and values, the lidar intensity scale, the absence of every id
        in hash_repr and str, the count marker in str, and the equality.
        """
        database = _database(self.corpus, self.cache_path)

        fields = database.hash_fields
        self.assertEqual(
            list(fields),
            [
                "database",
                "version",
                "box3d_pipelines",
                "lidar_pointcloud_num_features",
                "scenarios",
                "observed_steps",
                "dataset_format",
            ],
        )
        self.assertEqual(fields["database"], "AV2MotionDatabase")
        self.assertEqual(fields["box3d_pipelines"], "None")
        self.assertEqual(fields["lidar_pointcloud_num_features"], "None")
        self.assertEqual(fields["observed_steps"], "50")
        self.assertEqual(fields["dataset_format"], "av2")
        self.assertIn(f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)} sha256=", fields["scenarios"])
        self.assertIsNone(database.lidar_intensity_scale)
        for scenario_id in _TRAIN_IDS + _VAL_IDS:
            self.assertNotIn(scenario_id, database.hash_repr)
            self.assertNotIn(scenario_id, str(database))
        self.assertIn(f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)}", str(database))
        self.assertEqual(database, _database(self.corpus, self.cache_path))

    def test_hash_ignores_filesystem_locations(self) -> None:
        """
        Input: the same corpus under another root with another cache path.
        Expected: the records store paths relative to the root, so where the corpus and the
        cache live does not key the cache.
        Check: the hashes are equal and the root path is absent from hash_repr.
        """
        reference = _database(self.corpus, self.cache_path).database_hash
        relocated = _write_empty_corpus(
            self.tmp_path / "elsewhere" / "av2", {"train": _TRAIN_IDS, "val": _VAL_IDS}
        )

        self.assertEqual(
            _database(relocated, self.tmp_path / "other_cache").database_hash, reference
        )
        self.assertNotIn(str(self.corpus), _database(self.corpus, self.cache_path).hash_repr)

    def test_hash_tracks_content_settings(self) -> None:
        """
        Input: the reference database next to one per changed setting: another version, a
        stride of 2, the train split only, one scenario fewer, a narrowed vocabulary and a
        patched OBSERVED_STEPS constant.
        Expected: every setting that changes the records changes the hash; the vocabulary
        does not, since the records store no labels.
        Check: the hash of every variant against the reference.
        """
        reference = _database(self.corpus, self.cache_path).database_hash

        self.assertNotEqual(
            _database(self.corpus, self.cache_path, version="v2").database_hash, reference
        )
        self.assertNotEqual(
            _database(self.corpus, self.cache_path, sample_steps=2).database_hash, reference
        )
        self.assertNotEqual(
            _database(self.corpus, self.cache_path, splits=("train",)).database_hash, reference
        )
        fewer = _write_empty_corpus(
            self.tmp_path / "fewer", {"train": _TRAIN_IDS[:-1], "val": _VAL_IDS}
        )
        self.assertNotEqual(_database(fewer, self.cache_path).database_hash, reference)
        narrowed = LabelVocabulary({"vehicle": "vehicle"})
        self.assertEqual(
            _database(self.corpus, self.cache_path, vocabulary=narrowed).database_hash,
            reference,
        )
        # The constant is read through the database module, which imported it by name
        with (
            patch.object(av2_motion_records_generator, "OBSERVED_STEPS", 30),
            patch.object(av2_motion_database, "OBSERVED_STEPS", 30),
        ):
            self.assertNotEqual(_database(self.corpus, self.cache_path).database_hash, reference)

    def test_hash_is_memoized_and_stays_small_for_many_scenarios(self) -> None:
        """
        Input: a train split of 2000 empty scenario directories.
        Expected: the representation digests the ids instead of listing them, and the hash is
        computed once.
        Check: the length of hash_repr and the identity of two database_hash reads.
        """
        many = _write_empty_corpus(
            self.tmp_path / "many", {"train": [f"{index:05d}" for index in range(2000)]}
        )
        database = _database(many, self.cache_path, splits=("train",))

        self.assertLess(len(database.hash_repr), 2000)
        self.assertIs(database.database_hash, database.database_hash)


class TestAV2MotionDatabaseRecords(AV2MotionDatabaseTestCase):
    """The database writes one record per scenario into its cache file."""

    @unittest.skipUnless(_AV2_AVAILABLE, "the av2 devkit is not installed")
    def test_process_scenario_records_writes_one_record_per_scenario(self) -> None:
        """
        Input: three train, one val and one 50 step test scenario written as real files,
        processed by two workers with two scenarios per task.
        Expected: the cache file holds one record per scenario in split and id order, every
        record carries a motion struct with its relative tracks path and step count and
        no sensor annotations, and a second run finds the cache and leaves it untouched.
        Check: the cache file, the record ids and motion fields, the empty lidar frames
        and absent boxes, and the modification time of the cache after the second run.
        """
        root = self.tmp_path / "records"
        for scenario_id in ("a0", "a1", "a2"):
            write_scenario(root / "train", scenario_id)
        write_scenario(root / "val", "b0")
        write_scenario(root / "test", "c0", num_steps=50)
        database = _database(
            root,
            self.cache_path,
            splits=("train", "val", "test"),
            num_workers=2,
            scenarios_per_worker_task=2,
        )

        database.process_scenario_records()

        self.assertTrue(database.cache_file_path.is_file())
        records = database.load_scenario_records()
        self.assertEqual([record.scenario_id for record in records], ["a0", "a1", "a2", "b0", "c0"])
        self.assertTrue(all(record.motion is not None for record in records))
        self.assertEqual(records[0].motion.tracks_relative_path, "train/a0/scenario_a0.parquet")
        self.assertEqual(records[-1].motion.num_steps, 50)
        self.assertEqual(records[0].lidar_frames, [])
        self.assertIsNone(records[0].boxes_3d)

        modified = database.cache_file_path.stat().st_mtime_ns
        database.process_scenario_records()
        self.assertEqual(database.cache_file_path.stat().st_mtime_ns, modified)
