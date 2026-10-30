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

from pathlib import Path

import pytest

import autoware_ml.configs.paths as paths
from autoware_ml.configs.paths import (
    CONFIGS_ROOT,
    PROJECT_ROOT,
    SETTINGS_KEYS,
    data_path,
    read_settings,
    scenarios_path,
    working_dir,
)
from autoware_ml.configs.tests.settings import write_settings


def test_project_root_is_the_checkout_holding_the_package() -> None:
    assert PROJECT_ROOT.is_absolute()
    assert (PROJECT_ROOT / "autoware_ml" / "configs").resolve() == CONFIGS_ROOT


def test_the_checkout_settings_file_sets_every_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "SETTINGS_FILE", PROJECT_ROOT / ".env")

    assert read_settings().keys() == SETTINGS_KEYS


def test_read_settings_skips_comments_and_keeps_equal_signs_in_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_file = tmp_path / ".env"
    settings_file.write_text(
        "# Dataset root\n"
        "AUTOWARE_ML_DATA_PATH=/data/a=b\n"
        "\n"
        "AUTOWARE_ML_SCENARIOS_PATH=\n"
        "AUTOWARE_ML_WORKING_DIR=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "SETTINGS_FILE", settings_file)

    settings = read_settings()

    assert settings["AUTOWARE_ML_DATA_PATH"] == "/data/a=b"
    assert settings["AUTOWARE_ML_SCENARIOS_PATH"] == ""


@pytest.mark.parametrize(
    "content",
    [
        "AUTOWARE_ML_DATA_PATH=\nAUTOWARE_ML_SCENARIOS_PATH=\n",
        "AUTOWARE_ML_DATA_PATH=\nAUTOWARE_ML_SCENARIOS_PATH=\nAUTOWARE_ML_WORKING_DIR=\n"
        "AUTOWARE_ML_CUSTOM_MOUNTS=\n",
        "AUTOWARE_ML_DATA_PATH=/a\nAUTOWARE_ML_DATA_PATH=/b\nAUTOWARE_ML_SCENARIOS_PATH=\n"
        "AUTOWARE_ML_WORKING_DIR=\n",
    ],
    ids=["missing", "unknown", "repeated"],
)
def test_read_settings_rejects_other_keys(
    content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_file = tmp_path / ".env"
    settings_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr(paths, "SETTINGS_FILE", settings_file)

    with pytest.raises(ValueError, match="must set each of"):
        read_settings()


def test_read_settings_rejects_a_line_without_a_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_settings(tmp_path, monkeypatch)
    with open(tmp_path / ".env", "a", encoding="utf-8") as settings_file:
        settings_file.write("AUTOWARE_ML_DATA_PATH\n")

    with pytest.raises(ValueError, match="no KEY=value pair"):
        read_settings()


def test_data_path_requires_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_settings(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="AUTOWARE_ML_DATA_PATH is not set"):
        data_path()


@pytest.mark.parametrize("value", ["external/scenarios", "~/scenarios"], ids=["relative", "home"])
def test_read_settings_rejects_paths_that_are_not_absolute(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_settings(tmp_path, monkeypatch, AUTOWARE_ML_SCENARIOS_PATH=value)

    with pytest.raises(ValueError, match="not absolute"):
        read_settings()


def test_paths_are_the_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_settings(
        tmp_path,
        monkeypatch,
        AUTOWARE_ML_DATA_PATH=str(tmp_path / "data"),
        AUTOWARE_ML_SCENARIOS_PATH=str(tmp_path / "scenarios"),
        AUTOWARE_ML_WORKING_DIR=str(tmp_path / "outputs"),
    )

    assert data_path() == tmp_path / "data"
    assert scenarios_path() == tmp_path / "scenarios"
    assert working_dir() == tmp_path / "outputs"


def test_unset_paths_mean_no_scenarios_root_and_the_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_settings(tmp_path, monkeypatch)

    assert scenarios_path() is None
    assert working_dir() == PROJECT_ROOT
