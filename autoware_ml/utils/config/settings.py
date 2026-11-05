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

from dataclasses import dataclass
from pathlib import Path
from typing import Self

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_ROOT = PROJECT_ROOT / "autoware_ml" / "configs"
SETTINGS_FILE = PROJECT_ROOT / ".env"
SETTINGS_KEYS = frozenset(
    {
        "AUTOWARE_ML_DATA_PATH",
        "AUTOWARE_ML_CACHE_PATH",
        "AUTOWARE_ML_CONFIGS_PATH",
        "AUTOWARE_ML_SCENARIOS_PATH",
        "AUTOWARE_ML_WORKING_DIR",
    }
)


@dataclass(frozen=True)
class Settings:
    """Settings of a repository checkout.

    Every setting is an absolute host path, which docker mounts at the same path inside the
    container. A setting that is not set is ``None``.

    Attributes:
        data_path: Dataset root, ``AUTOWARE_ML_DATA_PATH``.
        cache_path: Root of the cached record tables, ``AUTOWARE_ML_CACHE_PATH``.
        configs_path: External config root laid out like the bundled configs,
            ``AUTOWARE_ML_CONFIGS_PATH``.
        scenarios_path: Root of the scenario lists, ``AUTOWARE_ML_SCENARIOS_PATH``.
        working_dir: Root of the experiment outputs, ``AUTOWARE_ML_WORKING_DIR``.
    """

    data_path: Path | None = None
    cache_path: Path | None = None
    configs_path: Path | None = None
    scenarios_path: Path | None = None
    working_dir: Path | None = None

    @classmethod
    def read(cls, settings_file: Path) -> Self:
        """Read the settings from a ``.env`` file.

        The file holds one ``KEY=value`` line per setting, empty lines and ``#`` comment lines. It
        is the only source of the settings, the process environment is not read. An empty value
        leaves the setting unset.

        Args:
            settings_file: The ``.env`` file.

        Returns:
            The settings of the file.

        Raises:
            ValueError: If a line is no ``KEY=value`` pair, a key repeats, the keys differ from the
                known settings or a path is not absolute.
        """
        lines = [line.strip() for line in settings_file.read_text(encoding="utf-8").splitlines()]
        entries = [line for line in lines if line and not line.startswith("#")]
        malformed = [line for line in entries if "=" not in line]
        if malformed:
            raise ValueError(f"{settings_file} holds lines that are no KEY=value pair: {malformed}")

        values = dict(line.split("=", 1) for line in entries)
        if len(values) != len(entries) or values.keys() != SETTINGS_KEYS:
            raise ValueError(
                f"{settings_file} must set each of {sorted(SETTINGS_KEYS)} once, "
                f"got {[line.split('=', 1)[0] for line in entries]}"
            )
        relative = {
            key: value for key, value in values.items() if value and not Path(value).is_absolute()
        }
        if relative:
            raise ValueError(f"{settings_file} holds paths that are not absolute: {relative}")

        paths = {key: Path(value) if value else None for key, value in values.items()}
        return cls(
            data_path=paths["AUTOWARE_ML_DATA_PATH"],
            cache_path=paths["AUTOWARE_ML_CACHE_PATH"],
            configs_path=paths["AUTOWARE_ML_CONFIGS_PATH"],
            scenarios_path=paths["AUTOWARE_ML_SCENARIOS_PATH"],
            working_dir=paths["AUTOWARE_ML_WORKING_DIR"],
        )

    def resolve_data_path(self) -> str:
        """Return the dataset root.

        Returns:
            Absolute dataset root.

        Raises:
            ValueError: If ``AUTOWARE_ML_DATA_PATH`` is not set.
        """
        if self.data_path is None:
            raise ValueError("AUTOWARE_ML_DATA_PATH is not set in .env.")
        return str(self.data_path)

    def resolve_cache_path(self) -> str:
        """Return the root of the cached record tables.

        Returns:
            Absolute root of the cached record tables.

        Raises:
            ValueError: If ``AUTOWARE_ML_CACHE_PATH`` is not set.
        """
        if self.cache_path is None:
            raise ValueError("AUTOWARE_ML_CACHE_PATH is not set in .env.")
        return str(self.cache_path)

    def resolve_scenarios_path(self, relative_path: str) -> str:
        """Return a directory of scenario lists below the root of the scenario lists.

        Args:
            relative_path: Path relative to the root of the scenario lists, e.g. the lists of one
                task.

        Returns:
            Absolute path of the directory.

        Raises:
            ValueError: If ``AUTOWARE_ML_SCENARIOS_PATH`` is not set.
        """
        if self.scenarios_path is None:
            raise ValueError(
                f"AUTOWARE_ML_SCENARIOS_PATH is not set in .env, it holds {relative_path}."
            )
        return str(self.scenarios_path / relative_path)

    def resolve_working_dir(self) -> str:
        """Return the root of the experiment outputs.

        Returns:
            Absolute root of the experiment outputs, the project root when the setting is not set.
        """
        return str(self.working_dir or PROJECT_ROOT)
