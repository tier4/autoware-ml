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

"""Dataset sources a datamodule split is assembled from."""

from __future__ import annotations

from dataclasses import dataclass

from autoware_ml.databases.base_database import BaseDatabase


@dataclass(frozen=True)
class DatasetSource:
    """One database and the annotations it contributes to a split.

    The det3d and seg3d flags choose which annotations of the source are used. A disabled task
    keeps its field so samples from all sources still collate together: the boxes become an
    empty set, which the detection loss and metrics read as a frame without objects, and every
    point gets the ignore index.

    Attributes:
      database: Database providing the dataset records of the source.
      det3d: Whether the boxes of the source are used.
      seg3d: Whether the semantic masks of the source are used.
      repeat: How many times the frames of the source appear in one epoch.
    """

    database: BaseDatabase
    det3d: bool = True
    seg3d: bool = True
    repeat: int = 1

    def __post_init__(self) -> None:
        """Validate the source declaration."""
        if not isinstance(self.database, BaseDatabase):
            raise TypeError(
                f"A dataset source needs a database, got {type(self.database).__name__}."
            )
        if self.repeat < 1:
            raise ValueError(f"A dataset source repeat must be at least 1, got {self.repeat}.")
