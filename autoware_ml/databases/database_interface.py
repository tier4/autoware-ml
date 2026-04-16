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

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence, Protocol
from types import MappingProxyType

from autoware_ml.databases.scenarios import Scenarios, ScenarioData
from autoware_ml.databases.schemas import DatasetRecord


class DatabaseInterface(Protocol):
    """Protocol for database classes that defines the common interface for every dataset type."""

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
          str: String representation of the database.
        """

        raise NotImplementedError("Database must define __str__!")

    @abstractmethod
    def __hash__(self) -> int:
        """
        Hash the database by its version and scenario IDs.

        Returns:
          int: Hash of the database.
        """

        raise NotImplementedError("Database must define __hash__!")

    @abstractmethod
    def __eq__(self, other: DatabaseInterface) -> bool:
        """
        Compare two databases by their version and scenario IDs.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """

        raise NotImplementedError("Database must define __eq__!")

    @property
    @abstractmethod
    def database_version(self) -> str:
        """
        Get the version of the database.

        Returns:
          str: Version of the database.
        """

        raise NotImplementedError("Database must define database_version!")

    @property
    @abstractmethod
    def scenarios(self) -> MappingProxyType[str, Scenarios]:
        """
        Get the scenarios for each scenario group.

        Returns:
          MappingProxyType[str, Scenarios]: Dictionary of scenario group name to scenarios.
        """

        raise NotImplementedError("Database must define scenarios!")

    @abstractmethod
    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]:
        """
        Get all scenario data from all scenario groups and keep their order the same.

        Returns:
          MappingProxyType[str, ScenarioData]: Dictionary of scenario ID to scenario data.
        """

        raise NotImplementedError("Database must define get_unique_scenario_data!")

    @abstractmethod
    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """
        Load scenario records from the database.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        raise NotImplementedError("Database must define load_scenario_records!")

    @abstractmethod
    def process_scenario_records(self) -> Sequence[DatasetRecord]:
        """
        Process scenario records from the database.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        raise NotImplementedError("Subclasses must define process_scenario_records method!")
