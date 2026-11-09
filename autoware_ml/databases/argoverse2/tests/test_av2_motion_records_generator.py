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

"""Tests for the Argoverse 2 motion forecasting records generator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("av2") is None:
    raise unittest.SkipTest("the av2 devkit is not installed")

from autoware_ml.databases.argoverse2.av2_motion_records_generator import (  # noqa: E402
    OBSERVED_STEPS,
    AV2MotionRecordsGenerator,
)
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (  # noqa: E402
    CITY,
    FOCAL_TRACK_ID,
    write_scenario,
)
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData  # noqa: E402
from autoware_ml.types.dataset import DatasetFormat  # noqa: E402


def _scenario_data(scenario_id: str, split: str = "train") -> ScenarioData:
    """Scenario data of one scenario of the split, with the split as its version."""
    return ScenarioData(
        dataset_params=DatasetParams(dataset_name=split, sample_steps=1),
        scenario_id=scenario_id,
        scenario_version=split,
    )


class TestAV2MotionRecordsGenerator(unittest.TestCase):
    """The generator turns one scenario directory into one motion record."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.root = Path(self._tmp_dir.name) / "av2"

    def test_one_record_per_scenario_with_relative_paths(self) -> None:
        """
        Input: the synthetic train scenario "abc" of 110 steps and three tracks.
        Expected: one record without sensor annotations, stamped at the last observed step,
        whose motion struct names the two files relative to the root, the focal track and
        the full window.
        Check: every basic field of the record, its timestamp, and every motion field.
        """
        write_scenario(self.root / "train", "abc")

        records = AV2MotionRecordsGenerator(
            self.root, _scenario_data("abc")
        ).generate_dataset_records()

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.scenario_id, "abc")
        self.assertEqual(record.sample_id, "abc")
        self.assertEqual(record.scenario_name, "abc")
        self.assertEqual(record.sample_index, 0)
        self.assertEqual(record.location, CITY)
        self.assertIsNone(record.vehicle_type)
        self.assertEqual(record.lidar_frames, [])
        self.assertIsNone(record.lidar_sources)
        self.assertIsNone(record.image_frames)
        self.assertIsNone(record.category_mapping)
        self.assertIsNone(record.boxes_3d)
        # The current step is the last observed one, at 1.0 s + 49 * 0.1 s
        self.assertAlmostEqual(record.timestamp_seconds, 1.0 + 0.1 * (OBSERVED_STEPS - 1))

        motion = record.motion
        self.assertIsNotNone(motion)
        self.assertEqual(motion.tracks_relative_path, "train/abc/scenario_abc.parquet")
        self.assertEqual(motion.map_relative_path, "train/abc/log_map_archive_abc.json")
        self.assertIs(motion.dataset_format, DatasetFormat.AV2)
        self.assertIsNone(motion.scenario_key)
        self.assertEqual(motion.focal_track_id, FOCAL_TRACK_ID)
        self.assertEqual(motion.first_step_index, 0)
        self.assertEqual(motion.current_step_index, OBSERVED_STEPS - 1)
        self.assertEqual(motion.num_steps, 110)
        self.assertAlmostEqual(motion.step_seconds, 0.1)
        self.assertEqual(motion.num_tracks, 3)

    def test_a_test_split_scenario_ends_at_the_current_step(self) -> None:
        """
        Input: a test split scenario of exactly the observed steps.
        Expected: the window is the observed steps and the current step is its last one.
        Check: num_steps, current_step_index and the tracks path under the test split.
        """
        write_scenario(self.root / "test", "t00", num_steps=OBSERVED_STEPS)

        (record,) = AV2MotionRecordsGenerator(
            self.root, _scenario_data("t00", split="test")
        ).generate_dataset_records()

        self.assertEqual(record.motion.num_steps, OBSERVED_STEPS)
        self.assertEqual(record.motion.current_step_index, OBSERVED_STEPS - 1)
        self.assertEqual(record.motion.tracks_relative_path, "test/t00/scenario_t00.parquet")

    def test_a_window_shorter_than_the_observed_steps_is_rejected(self) -> None:
        """
        Input: a scenario one step shorter than the observed steps.
        Expected: a scenario that cannot fill the observed window is refused.
        Check: ValueError naming the observed step count.
        """
        write_scenario(self.root / "train", "short", num_steps=OBSERVED_STEPS - 1)

        with self.assertRaisesRegex(ValueError, "fewer than the 50 observed steps"):
            AV2MotionRecordsGenerator(self.root, _scenario_data("short")).generate_dataset_records()

    def test_missing_files_are_rejected(self) -> None:
        """
        Input: a scenario directory that does not exist, and one whose map archive was
        removed.
        Expected: each of the two files is required, and the missing one is named.
        Check: ValueError naming the missing tracks file and the missing map file.
        """
        generator = AV2MotionRecordsGenerator(self.root, _scenario_data("none"))
        with self.assertRaisesRegex(ValueError, "missing its file .*scenario_none.parquet"):
            generator.generate_dataset_records()

        scenario_path = write_scenario(self.root / "train", "nomap")
        (scenario_path / "log_map_archive_nomap.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing its file .*log_map_archive_nomap.json"):
            AV2MotionRecordsGenerator(self.root, _scenario_data("nomap")).generate_dataset_records()

    def test_a_tracks_file_of_another_scenario_is_rejected(self) -> None:
        """
        Input: the files of scenario "abc" renamed into the directory of scenario "xyz".
        Expected: the scenario id stored inside the tracks file must match the directory, so
        a misplaced file is refused.
        Check: ValueError naming both ids.
        """
        scenario_path = write_scenario(self.root / "train", "abc")
        moved = self.root / "train" / "xyz"
        scenario_path.rename(moved)
        (moved / "scenario_abc.parquet").rename(moved / "scenario_xyz.parquet")
        (moved / "log_map_archive_abc.json").rename(moved / "log_map_archive_xyz.json")

        with self.assertRaisesRegex(ValueError, "belongs to scenario 'abc', not to 'xyz'"):
            AV2MotionRecordsGenerator(self.root, _scenario_data("xyz")).generate_dataset_records()
