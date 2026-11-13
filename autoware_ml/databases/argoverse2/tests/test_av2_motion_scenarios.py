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

import tempfile
import unittest
from pathlib import Path

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
    """Build the scenarios of the corpus, for the train and val splits unless given."""
    return AV2MotionScenarios(
        scenario_root_path=root,
        dataset_params=dataset_params
        or [
            DatasetParams(dataset_name="train", sample_steps=1),
            DatasetParams(dataset_name="val", sample_steps=1),
        ],
    )


class AV2MotionScenariosTestCase(unittest.TestCase):
    """Base case that writes a corpus of train and val scenario directories per test."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name)
        self.corpus = _write_corpus(
            self.tmp_path / "corpus", {"train": _TRAIN_IDS, "val": _VAL_IDS}
        )


class TestAV2MotionScenariosDiscovery(AV2MotionScenariosTestCase):
    """The scenarios are the directories of the requested split directories."""

    def test_scenarios_are_discovered_from_the_split_directories(self) -> None:
        """
        Input: a corpus with five train and two val scenario directories, requested with
        sample_steps 1 for both splits.
        Expected: every split directory becomes a split of scenario data, sorted by id, with
        the split name as the scenario version and no vehicle type or location.
        Check: the split keys, the ids per split in order, the fields of the first train
        scenario and the total count of scenario data.
        """
        scenarios = _scenarios(self.corpus)

        self.assertEqual(set(scenarios.scenario_data), {SplitType.TRAIN, SplitType.VAL})
        train = scenarios.scenario_data[SplitType.TRAIN]
        self.assertEqual([scenario.scenario_id for scenario in train], sorted(_TRAIN_IDS))
        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios.scenario_data[SplitType.VAL]],
            _VAL_IDS,
        )
        self.assertEqual(train[0].scenario_version, "train")
        self.assertEqual(train[0].dataset_params.dataset_name, "train")
        self.assertIsNone(train[0].vehicle_type)
        self.assertIsNone(train[0].location)
        self.assertEqual(len(scenarios.get_all_scenario_data()), len(_TRAIN_IDS) + len(_VAL_IDS))

    def test_files_in_a_split_directory_are_not_scenarios(self) -> None:
        """
        Input: the corpus with a README file next to the train scenario directories.
        Expected: only directories are scenarios, so the file is skipped.
        Check: the train split still holds exactly the five scenario directories.
        """
        (self.corpus / "train" / "README.md").write_text("not a scenario")

        scenarios = _scenarios(self.corpus)

        self.assertEqual(len(scenarios.scenario_data[SplitType.TRAIN]), len(_TRAIN_IDS))

    def test_sample_steps_strides_over_the_sorted_scenarios(self) -> None:
        """
        Input: the train split requested with sample_steps 2.
        Expected: every second scenario of the sorted ids is kept, starting with the first.
        Check: the kept ids equal the sorted ids with stride 2.
        """
        scenarios = _scenarios(self.corpus, [DatasetParams(dataset_name="train", sample_steps=2)])

        kept = [scenario.scenario_id for scenario in scenarios.scenario_data[SplitType.TRAIN]]
        self.assertEqual(kept, sorted(_TRAIN_IDS)[::2])
        self.assertTrue(kept[0].startswith("0a1b"))

    def test_a_test_split_is_served_as_the_test_split(self) -> None:
        """
        Input: the corpus with a test split directory of one scenario, requested alone.
        Expected: the test directory maps to the test split type.
        Check: the split keys of the scenario data.
        """
        _write_corpus(self.corpus, {"test": ["t000"]})

        scenarios = _scenarios(self.corpus, [DatasetParams(dataset_name="test", sample_steps=1)])

        self.assertEqual(list(scenarios.scenario_data), [SplitType.TEST])


class TestAV2MotionScenariosValidation(AV2MotionScenariosTestCase):
    """A split request that cannot be served is refused."""

    def test_an_unknown_split_name_is_rejected(self) -> None:
        """
        Input: a dataset name that is no Argoverse 2 motion forecasting split.
        Expected: the split is refused by name before any directory is read.
        Check: ValueError naming the unknown split.
        """
        with self.assertRaisesRegex(
            ValueError, "Unknown Argoverse 2 motion forecasting split 'mini'"
        ):
            _scenarios(self.corpus, [DatasetParams(dataset_name="mini", sample_steps=1)])

    def test_a_missing_split_directory_is_rejected(self) -> None:
        """
        Input: the test split requested from a corpus that holds no test directory.
        Expected: a known split whose directory is missing is refused instead of yielding an
        empty split.
        Check: ValueError naming the missing split directory.
        """
        with self.assertRaisesRegex(ValueError, "split directory .*test.* does not exist"):
            _scenarios(self.corpus, [DatasetParams(dataset_name="test", sample_steps=1)])

    def test_a_split_listed_twice_is_rejected(self) -> None:
        """
        Input: the train split requested twice with different sample steps.
        Expected: a split appears once in the request, so the duplicate is refused.
        Check: ValueError saying the split is listed more than once.
        """
        with self.assertRaisesRegex(ValueError, "listed more than once"):
            _scenarios(
                self.corpus,
                [
                    DatasetParams(dataset_name="train", sample_steps=1),
                    DatasetParams(dataset_name="train", sample_steps=2),
                ],
            )


class TestAV2MotionScenariosRepresentations(AV2MotionScenariosTestCase):
    """The representations name counts and digests, never the ids of the scenarios."""

    def test_representations_stay_compact(self) -> None:
        """
        Input: the scenarios of the corpus.
        Expected: every representation names counts, never the ids of up to 200k scenarios,
        and two scenarios objects of the same corpus hash alike.
        Check: the count and digest markers in str and hash_repr, the absence of every id in
        both, and the equality of the hashes.
        """
        scenarios = _scenarios(self.corpus)

        self.assertIn(f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)}", str(scenarios))
        self.assertIn(f"{SplitType.TRAIN}: n={len(_TRAIN_IDS)} sha256=", scenarios.hash_repr)
        for scenario_id in _TRAIN_IDS + _VAL_IDS:
            self.assertNotIn(scenario_id, str(scenarios))
            self.assertNotIn(scenario_id, scenarios.hash_repr)
        self.assertEqual(hash(scenarios), hash(_scenarios(self.corpus)))

    def test_hash_repr_ignores_the_root_path_and_tracks_the_ids(self) -> None:
        """
        Input: the same corpus under another root, a corpus with one train id renamed, a
        corpus with one train id fewer, and the original corpus with sample_steps 2 on train.
        Expected: the content representation does not depend on where the corpus lives, but
        on which scenarios it holds and on the stride that selects them.
        Check: the root path is absent from hash_repr, the relocated corpus hashes the same,
        the renamed, the smaller and the strided ones hash differently, and the stride is
        named in hash_repr.
        """
        reference = _scenarios(self.corpus).hash_repr
        self.assertNotIn(str(self.corpus), reference)

        relocated = _write_corpus(
            self.tmp_path / "elsewhere", {"train": _TRAIN_IDS, "val": _VAL_IDS}
        )
        self.assertEqual(_scenarios(relocated).hash_repr, reference)

        renamed = _write_corpus(
            self.tmp_path / "renamed", {"train": _TRAIN_IDS[:-1] + ["zzzz"], "val": _VAL_IDS}
        )
        self.assertNotEqual(_scenarios(renamed).hash_repr, reference)

        fewer = _write_corpus(self.tmp_path / "fewer", {"train": _TRAIN_IDS[:-1], "val": _VAL_IDS})
        self.assertNotEqual(_scenarios(fewer).hash_repr, reference)

        strided = _scenarios(
            self.corpus,
            [
                DatasetParams(dataset_name="train", sample_steps=2),
                DatasetParams(dataset_name="val", sample_steps=1),
            ],
        )
        self.assertNotEqual(strided.hash_repr, reference)
        self.assertIn("sample_steps=2", strided.hash_repr)
