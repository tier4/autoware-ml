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

"""Blob paths of the record tables, stored relative to the database root.

A record table names its point clouds, masks, source indexes and images relative to the root
of the database that generated it. Each records generator strips its own root, whatever the
layout of its database, and the dataset joins the stored path onto its own root when it reads
the record back. The records therefore stay valid when the database is mounted elsewhere, and
the dataset needs no knowledge of the directory layout below the root.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def relative_to_database_root(path: str | Path, database_root_path: str | Path) -> str:
    """
    Strip the database root off a path a devkit resolved under it.

    Args:
      path: Path of a file under the database root.
      database_root_path: Root directory of the database.

    Returns:
      str: The path relative to the database root, with POSIX separators.

    Raises:
      ValueError: If the path does not lie under the database root.
    """

    try:
        return Path(path).relative_to(database_root_path).as_posix()
    except ValueError as error:
        raise ValueError(
            f"File {path} does not lie under the database root {database_root_path}, so it "
            "cannot be stored relative to it."
        ) from error


def check_relative_path(path: str, field_name: str) -> str:
    """
    Check that a stored blob path is relative, as the record tables require.

    Args:
      path: Stored blob path.
      field_name: Name of the field with the path, used in the error message.

    Returns:
      str: The path, unchanged.

    Raises:
      ValueError: If the path is empty or absolute.
    """

    if path == "":
        raise ValueError(f"{field_name} is empty, expected a path relative to the database root.")
    if PurePosixPath(path).is_absolute():
        raise ValueError(
            f"{field_name} '{path}' is absolute, expected a path relative to the database root."
        )
    return path
