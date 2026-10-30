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

"""Settings files the tests point autoware-ml at instead of the ``.env`` of the checkout."""

from pathlib import Path

import pytest

import autoware_ml.configs.paths as paths


def write_settings(directory: Path, monkeypatch: pytest.MonkeyPatch, **values: str) -> Path:
    """Write a ``.env`` with the given settings, every other one not set, and read it instead.

    Args:
        directory: Directory the file is written to.
        monkeypatch: Fixture that points the settings at the file for the test.
        **values: Setting values by key.

    Returns:
        Path of the written file.
    """
    settings = {key: "" for key in paths.SETTINGS_KEYS} | values
    settings_file = directory / ".env"
    settings_file.write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(settings.items())), encoding="utf-8"
    )
    monkeypatch.setattr(paths, "SETTINGS_FILE", settings_file)
    return settings_file
