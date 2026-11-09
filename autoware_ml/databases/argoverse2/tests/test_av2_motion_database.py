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

from pathlib import Path
from unittest.mock import patch

import pytest

from autoware_ml.databases.argoverse2 import av2_motion_database, av2_motion_records_generator
from autoware_ml.databases.argoverse2.av2_motion_database import AV2MotionDatabase
from autoware_ml.databases.argoverse2.av2_motion_scenarios import AV2MotionScenarios
from autoware_ml.databases.argoverse2.tests.av2_fixtures import av2_vocabulary, write_scenario
from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.taxonomy import LabelVocabulary
from autoware_ml.types.dataset import SplitType

_TRAIN_IDS = [f"{index:04d}" for index in range(6)]
_VAL_IDS = ["v000", "v001"]


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


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _write_empty_corpus(tmp_path / "av2", {"train": _TRAIN_IDS, "val": _VAL_IDS})


def test_rejects_an_empty_worker_task(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenarios_per_worker_task must be positive"):
        _database(corpus, tmp_path / "cache", scenarios_per_worker_task=0)


def test_hash_fields_and_compact_representations(corpus: Path, tmp_path: Path) -> None:
    database = _database(corpus, tmp_path / "cache")

    fields = database.hash_fields
    assert list(fields) == [
        "database",
        "version",
        "box3d_pipelines",
        "lidar_pointcloud_num_features",
        "scenarios",
        "observed_steps",
        "dataset_format",
    ]
    assert fields["database"] == "AV2MotionDatabase"
    assert fields["box3d_pipelines"] == "None"
    assert fields["lidar_pointcloud_num_features"] == "None"
    assert fields["observed_steps"] == "50"
    assert fields["dataset_format"] == "av2"
    assert f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)} sha256=" in fields["scenarios"]
    assert database.lidar_intensity_scale is None
    # Neither representation lists the scenario ids
    for scenario_id in _TRAIN_IDS + _VAL_IDS:
        assert scenario_id not in database.hash_repr
        assert scenario_id not in str(database)
    assert f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)}" in str(database)
    assert database == _database(corpus, tmp_path / "cache")


def test_hash_ignores_filesystem_locations(corpus: Path, tmp_path: Path) -> None:
    reference = _database(corpus, tmp_path / "cache").database_hash
    relocated = _write_empty_corpus(
        tmp_path / "elsewhere" / "av2", {"train": _TRAIN_IDS, "val": _VAL_IDS}
    )

    assert _database(relocated, tmp_path / "other_cache").database_hash == reference
    assert str(corpus) not in _database(corpus, tmp_path / "cache").hash_repr


def test_hash_tracks_content_settings(corpus: Path, tmp_path: Path) -> None:
    reference = _database(corpus, tmp_path / "cache").database_hash

    assert _database(corpus, tmp_path / "cache", version="v2").database_hash != reference
    assert _database(corpus, tmp_path / "cache", sample_steps=2).database_hash != reference
    assert _database(corpus, tmp_path / "cache", splits=("train",)).database_hash != reference
    fewer = _write_empty_corpus(tmp_path / "fewer", {"train": _TRAIN_IDS[:-1], "val": _VAL_IDS})
    assert _database(fewer, tmp_path / "cache").database_hash != reference
    # The records store no labels, so the vocabulary does not key the cache
    narrowed = LabelVocabulary({"vehicle": "vehicle"})
    assert _database(corpus, tmp_path / "cache", vocabulary=narrowed).database_hash == reference
    # The constant is read through the database module, which imported it by name
    with (
        patch.object(av2_motion_records_generator, "OBSERVED_STEPS", 30),
        patch.object(av2_motion_database, "OBSERVED_STEPS", 30),
    ):
        assert _database(corpus, tmp_path / "cache").database_hash != reference


def test_hash_is_memoized_and_stays_small_for_many_scenarios(tmp_path: Path) -> None:
    many = _write_empty_corpus(
        tmp_path / "many", {"train": [f"{index:05d}" for index in range(2000)]}
    )
    database = _database(many, tmp_path / "cache", splits=("train",))

    assert len(database.hash_repr) < 2000
    assert database.database_hash is database.database_hash


def test_process_scenario_records_writes_one_record_per_scenario(tmp_path: Path) -> None:
    pytest.importorskip("av2")
    root = tmp_path / "av2"
    for scenario_id in ("a0", "a1", "a2"):
        write_scenario(root / "train", scenario_id)
    write_scenario(root / "val", "b0")
    write_scenario(root / "test", "c0", num_steps=50)
    database = _database(
        root,
        tmp_path / "cache",
        splits=("train", "val", "test"),
        num_workers=2,
        scenarios_per_worker_task=2,
    )

    database.process_scenario_records()

    assert database.cache_file_path.is_file()
    records = database.load_scenario_records()
    assert [record.scenario_id for record in records] == ["a0", "a1", "a2", "b0", "c0"]
    assert all(record.prediction is not None for record in records)
    assert records[0].prediction.tracks_relative_path == "train/a0/scenario_a0.parquet"
    assert records[-1].prediction.num_steps == 50
    assert records[0].lidar_frames == []
    assert records[0].boxes_3d is None

    # A second run finds the cache and leaves it untouched
    modified = database.cache_file_path.stat().st_mtime_ns
    database.process_scenario_records()
    assert database.cache_file_path.stat().st_mtime_ns == modified
