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

"""Filesystem paths of the bundled configs and the settings of the repository checkout."""

from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = PROJECT_ROOT / ".env"
SETTINGS_KEYS = frozenset(
    {
        "AUTOWARE_ML_DATA_PATH",
        "AUTOWARE_ML_SCENARIOS_PATH",
        "AUTOWARE_ML_WORKING_DIR",
    }
)


def read_settings() -> Mapping[str, str]:
    """Read the settings of the checkout from ``.env`` at the project root.

    The file holds one ``KEY=value`` line per setting, empty lines and ``#`` comment lines. It is
    the only source of the settings, the process environment is not read. Every setting is an
    absolute host path, which docker mounts at the same path inside the container.

    Returns:
        Value of every setting, an empty string for a setting that is not set.

    Raises:
        ValueError: If a line is no ``KEY=value`` pair, a key repeats, the keys differ from the
            known settings or a path is not absolute.
    """
    lines = [line.strip() for line in SETTINGS_FILE.read_text(encoding="utf-8").splitlines()]
    entries = [line for line in lines if line and not line.startswith("#")]
    malformed = [line for line in entries if "=" not in line]
    if malformed:
        raise ValueError(f"{SETTINGS_FILE} holds lines that are no KEY=value pair: {malformed}")

    settings = dict(line.split("=", 1) for line in entries)
    if len(settings) != len(entries) or settings.keys() != SETTINGS_KEYS:
        raise ValueError(
            f"{SETTINGS_FILE} must set each of {sorted(SETTINGS_KEYS)} once, "
            f"got {[line.split('=', 1)[0] for line in entries]}"
        )
    relative = {
        key: value for key, value in settings.items() if value and not Path(value).is_absolute()
    }
    if relative:
        raise ValueError(f"{SETTINGS_FILE} holds paths that are not absolute: {relative}")
    return settings


def data_path() -> Path:
    """Return the dataset root named by ``AUTOWARE_ML_DATA_PATH``.

    Returns:
        Absolute dataset root.

    Raises:
        ValueError: If ``AUTOWARE_ML_DATA_PATH`` is not set.
    """
    value = read_settings()["AUTOWARE_ML_DATA_PATH"]
    if not value:
        raise ValueError(f"AUTOWARE_ML_DATA_PATH is not set in {SETTINGS_FILE}.")
    return Path(value)


def scenarios_path() -> Path | None:
    """Return the root of the scenario lists named by ``AUTOWARE_ML_SCENARIOS_PATH``.

    The T4Dataset configs read their scenario lists from ``t4dataset/`` below the root.

    Returns:
        Absolute root of the scenario lists, ``None`` when the setting is not set.
    """
    value = read_settings()["AUTOWARE_ML_SCENARIOS_PATH"]
    return Path(value) if value else None


def working_dir() -> Path:
    """Return the root of the experiment outputs named by ``AUTOWARE_ML_WORKING_DIR``.

    Returns:
        Absolute root of the experiment outputs, the project root when the setting is not set.
    """
    value = read_settings()["AUTOWARE_ML_WORKING_DIR"]
    return Path(value) if value else PROJECT_ROOT
