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

"""Unit tests for the T4 scenario ID parser, in particular the traffic cone / barrier
annotation flag carried as the fifth segment of a scenario ID."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.types.dataset import SplitType

_DATASET_NAME = "db_test_v1"


def _dataset_params() -> DatasetParams:
    """Build the dataset parameters every parsed scenario is attached to."""
    return DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1)


def _parse(scenario_id: str) -> ScenarioData:
    """Parse one scenario ID as it appears in a split yaml."""
    return T4Scenarios._build_scenario_data(
        scenario_id=scenario_id, dataset_params=_dataset_params()
    )


class TestTrafficConeBarrierBboxStatus(unittest.TestCase):
    """The fifth segment of a scenario ID says whether cones and barriers are annotated."""

    def test_true_marks_the_scenario_as_annotated(self) -> None:
        """``true`` yields the boolean flag alongside the location and vehicle type."""
        scenario = _parse("scene_a/0/tokyo/j6gen2/true")

        self.assertIs(scenario.traffic_cone_barrier_bbox_status, True)
        self.assertEqual(scenario.scenario_id, "scene_a")
        self.assertEqual(scenario.scenario_version, "0")
        self.assertEqual(scenario.location, "tokyo")
        self.assertEqual(scenario.vehicle_type, "j6gen2")

    def test_false_marks_the_scenario_as_not_annotated(self) -> None:
        """``false``, the value most split yamls carry, yields ``False`` (not ``bool("false")``)."""
        scenario = _parse("scene_a/0/tokyo/j6gen2/false")

        self.assertIs(scenario.traffic_cone_barrier_bbox_status, False)

    def test_none_marks_the_status_as_unknown(self) -> None:
        """``none`` keeps the location and vehicle type but leaves the flag unknown."""
        scenario = _parse("scene_a/0/tokyo/j6gen2/none")

        self.assertIsNone(scenario.traffic_cone_barrier_bbox_status)
        self.assertEqual(scenario.location, "tokyo")
        self.assertEqual(scenario.vehicle_type, "j6gen2")

    def test_flag_text_is_case_insensitive(self) -> None:
        """``True`` / ``FALSE`` spellings parse like their lowercase forms."""
        self.assertIs(_parse("scene_a/0/tokyo/j6gen2/True").traffic_cone_barrier_bbox_status, True)
        self.assertIs(
            _parse("scene_a/0/tokyo/j6gen2/FALSE").traffic_cone_barrier_bbox_status, False
        )

    def test_absent_flag_defaults_to_unknown(self) -> None:
        """A two-segment ID carries no metadata, so the flag is ``None`` rather than a guess."""
        scenario = _parse("scene_a/0")

        self.assertIsNone(scenario.traffic_cone_barrier_bbox_status)
        self.assertEqual(scenario.scenario_id, "scene_a")
        self.assertEqual(scenario.scenario_version, "0")
        self.assertIsNone(scenario.location)
        self.assertIsNone(scenario.vehicle_type)

    def test_unrecognised_flag_values_are_rejected(self) -> None:
        """Anything but ``true``, ``false`` or ``none`` is a typo and must not become a flag."""
        for value in ("maybe", "yes", "1", ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "traffic cone"):
                    _parse(f"scene_a/0/tokyo/j6gen2/{value}")

    def test_other_segment_counts_are_rejected(self) -> None:
        """Only the bare and the fully qualified forms are scenario IDs."""
        for scenario_id in ("scene_a", "scene_a/0/tokyo", "scene_a/0/tokyo/j6gen2/true/extra"):
            with self.subTest(scenario_id=scenario_id):
                with self.assertRaisesRegex(ValueError, "Invalid scenario ID"):
                    _parse(scenario_id)


class TestScenarioYaml(unittest.TestCase):
    """The split yaml is parsed into per-split scenario data carrying the flag."""

    def setUp(self) -> None:
        """Write a split yaml mixing fully qualified and bare scenario IDs."""
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.scenario_root_path = Path(self._tmp_dir.name)
        splits = {
            "train": ["scene_a/0/tokyo/j6gen2/true", "scene_b/1"],
            "val": ["scene_c/2/odaiba/jpntaxi/true"],
            "test": [],
        }
        with open(self.scenario_root_path / f"{_DATASET_NAME}.yaml", "w") as f:
            yaml.safe_dump(splits, f)

    def test_flag_is_carried_per_scenario(self) -> None:
        """Each scenario keeps its own flag, and bare IDs stay unknown."""
        scenarios = T4Scenarios(
            scenario_root_path=self.scenario_root_path, dataset_params=[_dataset_params()]
        )

        train = scenarios.scenario_data[SplitType.TRAIN]
        val = scenarios.scenario_data[SplitType.VAL]
        self.assertEqual([scenario.scenario_id for scenario in train], ["scene_a", "scene_b"])
        self.assertIs(train[0].traffic_cone_barrier_bbox_status, True)
        self.assertIsNone(train[1].traffic_cone_barrier_bbox_status)
        self.assertEqual(len(val), 1)
        self.assertIs(val[0].traffic_cone_barrier_bbox_status, True)
        self.assertEqual(val[0].location, "odaiba")
        self.assertEqual(scenarios.scenario_data[SplitType.TEST], [])

    def test_flag_takes_part_in_scenario_equality(self) -> None:
        """Two scenarios differing only in the flag are different scenarios."""
        annotated = _parse("scene_a/0/tokyo/j6gen2/true")
        unknown = ScenarioData(
            dataset_params=_dataset_params(),
            scenario_id="scene_a",
            scenario_version="0",
            vehicle_type="j6gen2",
            location="tokyo",
            traffic_cone_barrier_bbox_status=None,
        )

        self.assertNotEqual(annotated, unknown)


if __name__ == "__main__":
    unittest.main()
