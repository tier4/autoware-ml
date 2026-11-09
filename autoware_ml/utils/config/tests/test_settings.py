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

"""Tests for the settings of the checkout read from ``.env``."""

import tempfile
import unittest
from pathlib import Path

from autoware_ml.utils.config.settings import CONFIGS_ROOT, PROJECT_ROOT, SETTINGS_FILE, Settings

UNSET_SETTINGS = (
    "AUTOWARE_ML_DATA_PATH=\n"
    "AUTOWARE_ML_CACHE_PATH=\n"
    "AUTOWARE_ML_CONFIGS_PATH=\n"
    "AUTOWARE_ML_SCENARIOS_PATH=\n"
    "AUTOWARE_ML_WORKING_DIR=\n"
)


class TestSettingsRead(unittest.TestCase):
    """Tests for reading the settings from a ``.env`` file."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.settings_file = Path(directory.name) / ".env"

    def read(self, content: str) -> Settings:
        self.settings_file.write_text(content, encoding="utf-8")
        return Settings.read(self.settings_file)

    def test_the_checkout_settings_file_is_valid(self) -> None:
        Settings.read(SETTINGS_FILE)

    def test_reads_every_setting(self) -> None:
        settings = self.read(
            "AUTOWARE_ML_DATA_PATH=/data\n"
            "AUTOWARE_ML_CACHE_PATH=/cache\n"
            "AUTOWARE_ML_CONFIGS_PATH=/configs\n"
            "AUTOWARE_ML_SCENARIOS_PATH=/scenarios\n"
            "AUTOWARE_ML_WORKING_DIR=/outputs\n"
        )

        self.assertEqual(
            settings,
            Settings(
                data_path=Path("/data"),
                cache_path=Path("/cache"),
                configs_path=Path("/configs"),
                scenarios_path=Path("/scenarios"),
                working_dir=Path("/outputs"),
            ),
        )

    def test_empty_values_leave_the_settings_unset(self) -> None:
        self.assertEqual(self.read(UNSET_SETTINGS), Settings())

    def test_skips_comments_and_keeps_equal_signs_in_values(self) -> None:
        settings = self.read(
            "# Dataset root\n"
            "AUTOWARE_ML_DATA_PATH=/data/a=b\n"
            "\n"
            "AUTOWARE_ML_CACHE_PATH=\n"
            "AUTOWARE_ML_CONFIGS_PATH=\n"
            "AUTOWARE_ML_SCENARIOS_PATH=\n"
            "AUTOWARE_ML_WORKING_DIR=\n"
        )

        self.assertEqual(settings, Settings(data_path=Path("/data/a=b")))

    def test_rejects_other_keys(self) -> None:
        contents = {
            "missing": UNSET_SETTINGS.replace("AUTOWARE_ML_CACHE_PATH=\n", ""),
            "unknown": UNSET_SETTINGS + "AUTOWARE_ML_CUSTOM_MOUNTS=\n",
            "repeated": UNSET_SETTINGS + "AUTOWARE_ML_DATA_PATH=/data\n",
        }
        for case, content in contents.items():
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "must set each of"):
                self.read(content)

    def test_rejects_a_line_without_a_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "no KEY=value pair"):
            self.read(UNSET_SETTINGS + "AUTOWARE_ML_DATA_PATH\n")

    def test_rejects_paths_that_are_not_absolute(self) -> None:
        for value in ("external/scenarios", "~/scenarios"):
            content = UNSET_SETTINGS.replace(
                "AUTOWARE_ML_SCENARIOS_PATH=", f"AUTOWARE_ML_SCENARIOS_PATH={value}"
            )
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "not absolute"):
                self.read(content)


class TestSettingsResolve(unittest.TestCase):
    """Tests for the paths the settings resolve to."""

    def test_project_root_is_the_checkout_holding_the_configs(self) -> None:
        self.assertTrue(PROJECT_ROOT.is_absolute())
        self.assertTrue((CONFIGS_ROOT / "tasks").is_dir())

    def test_resolves_the_set_paths(self) -> None:
        settings = Settings(
            data_path=Path("/data"),
            cache_path=Path("/cache"),
            scenarios_path=Path("/scenarios"),
            working_dir=Path("/outputs"),
        )

        self.assertEqual(settings.resolve_data_path(), "/data")
        self.assertEqual(settings.resolve_cache_path(), "/cache")
        self.assertEqual(settings.resolve_scenarios_path("task"), "/scenarios/task")
        self.assertEqual(settings.resolve_working_dir(), "/outputs")

    def test_working_dir_defaults_to_the_project_root(self) -> None:
        self.assertEqual(Settings().resolve_working_dir(), str(PROJECT_ROOT))

    def test_data_path_requires_the_setting(self) -> None:
        with self.assertRaisesRegex(ValueError, "AUTOWARE_ML_DATA_PATH is not set"):
            Settings().resolve_data_path()

    def test_cache_path_defaults_to_the_cache_of_the_project_root(self) -> None:
        self.assertEqual(Settings().resolve_cache_path(), str(PROJECT_ROOT / ".cache"))

    def test_scenarios_path_requires_the_setting(self) -> None:
        with self.assertRaisesRegex(ValueError, "AUTOWARE_ML_SCENARIOS_PATH is not set"):
            Settings().resolve_scenarios_path("task")


if __name__ == "__main__":
    unittest.main()
