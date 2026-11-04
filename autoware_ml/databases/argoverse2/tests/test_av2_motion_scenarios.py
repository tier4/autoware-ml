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

"""Tests for the Argoverse 2 motion forecasting scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_ml.databases.argoverse2.av2_motion_scenarios import AV2MotionScenarios
from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.types.dataset import SplitType

_TRAIN_IDS = ["0a1b", "0a1c", "0b00", "1fff", "2abc"]
_VAL_IDS = ["9x00", "9x01"]


def _write_corpus(root: Path, splits: dict[str, list[str]]) -> Path:
    """Write empty scenario directories under the split directories."""
    for split, scenario_ids in splits.items():
        for scenario_id in scenario_ids:
            (root / split / scenario_id).mkdir(parents=True)
    return root


def _scenarios(root: Path, dataset_params: list[DatasetParams] | None = None) -> AV2MotionScenarios:
    return AV2MotionScenarios(
        scenario_root_path=root,
        dataset_params=dataset_params
        or [
            DatasetParams(dataset_name="train", sample_steps=1),
            DatasetParams(dataset_name="val", sample_steps=1),
        ],
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path / "corpus", {"train": _TRAIN_IDS, "val": _VAL_IDS})


def test_scenarios_are_discovered_from_the_split_directories(corpus: Path) -> None:
    scenarios = _scenarios(corpus)

    assert set(scenarios.scenario_data) == {SplitType.TRAIN, SplitType.VAL}
    train = scenarios.scenario_data[SplitType.TRAIN]
    assert [scenario.scenario_id for scenario in train] == sorted(_TRAIN_IDS)
    assert [scenario.scenario_id for scenario in scenarios.scenario_data[SplitType.VAL]] == _VAL_IDS
    assert train[0].scenario_version == "train"
    assert train[0].dataset_params.dataset_name == "train"
    assert train[0].vehicle_type is None
    assert train[0].location is None
    assert len(scenarios.get_all_scenario_data()) == len(_TRAIN_IDS) + len(_VAL_IDS)


def test_files_in_a_split_directory_are_not_scenarios(corpus: Path) -> None:
    (corpus / "train" / "README.md").write_text("not a scenario")

    scenarios = _scenarios(corpus)

    assert len(scenarios.scenario_data[SplitType.TRAIN]) == len(_TRAIN_IDS)


def test_sample_steps_strides_over_the_sorted_scenarios(corpus: Path) -> None:
    scenarios = _scenarios(corpus, [DatasetParams(dataset_name="train", sample_steps=2)])

    kept = [scenario.scenario_id for scenario in scenarios.scenario_data[SplitType.TRAIN]]
    assert kept == sorted(_TRAIN_IDS)[::2]
    assert kept[0].startswith("0a1b")


def test_a_test_split_is_served_as_the_test_split(corpus: Path) -> None:
    _write_corpus(corpus, {"test": ["t000"]})

    scenarios = _scenarios(corpus, [DatasetParams(dataset_name="test", sample_steps=1)])

    assert list(scenarios.scenario_data) == [SplitType.TEST]


def test_an_unknown_split_name_is_rejected(corpus: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Argoverse 2 motion forecasting split 'mini'"):
        _scenarios(corpus, [DatasetParams(dataset_name="mini", sample_steps=1)])


def test_a_missing_split_directory_is_rejected(corpus: Path) -> None:
    with pytest.raises(ValueError, match="split directory .*test.* does not exist"):
        _scenarios(corpus, [DatasetParams(dataset_name="test", sample_steps=1)])


def test_a_split_listed_twice_is_rejected(corpus: Path) -> None:
    with pytest.raises(ValueError, match="listed more than once"):
        _scenarios(
            corpus,
            [
                DatasetParams(dataset_name="train", sample_steps=1),
                DatasetParams(dataset_name="train", sample_steps=2),
            ],
        )


def test_representations_stay_compact(corpus: Path) -> None:
    # Every representation names counts, never the ids of up to 200k scenarios
    scenarios = _scenarios(corpus)

    assert f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)}" in str(scenarios)
    assert f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)} sha256=" in scenarios.hash_repr
    for scenario_id in _TRAIN_IDS + _VAL_IDS:
        assert scenario_id not in str(scenarios)
        assert scenario_id not in scenarios.hash_repr
    assert hash(scenarios) == hash(_scenarios(corpus))


def test_hash_repr_ignores_the_root_path_and_tracks_the_ids(corpus: Path, tmp_path: Path) -> None:
    reference = _scenarios(corpus).hash_repr
    assert str(corpus) not in reference

    relocated = _write_corpus(tmp_path / "elsewhere", {"train": _TRAIN_IDS, "val": _VAL_IDS})
    assert _scenarios(relocated).hash_repr == reference

    renamed = _write_corpus(
        tmp_path / "renamed", {"train": _TRAIN_IDS[:-1] + ["zzzz"], "val": _VAL_IDS}
    )
    assert _scenarios(renamed).hash_repr != reference

    fewer = _write_corpus(tmp_path / "fewer", {"train": _TRAIN_IDS[:-1], "val": _VAL_IDS})
    assert _scenarios(fewer).hash_repr != reference

    strided = _scenarios(
        corpus,
        [
            DatasetParams(dataset_name="train", sample_steps=2),
            DatasetParams(dataset_name="val", sample_steps=1),
        ],
    )
    assert strided.hash_repr != reference
    assert "sample_steps=2" in strided.hash_repr
