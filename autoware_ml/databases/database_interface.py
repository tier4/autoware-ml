from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Sequence, Protocol
from types import MappingProxyType

from autoware_ml.databases.scenarios import Scenarios, ScenarioData
from autoware_ml.databases.schemas import DatasetRecord


class DatabaseInterface(Protocol):
    """Interface for database classes."""

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the database."""
        raise NotImplementedError("Database must define __str__!")

    @abstractmethod
    def __hash__(self) -> int:
        """Hash the database by its version and scenario IDs."""
        raise NotImplementedError("Database must define __hash__!")

    @abstractmethod
    def __eq__(self, other: DatabaseInterface) -> bool:
        """Compare two databases by their version and scenario IDs."""
        raise NotImplementedError("Database must define __eq__!")

    @property
    @abstractmethod
    def cache_file_prefix_name(self) -> str:
        """Get the prefix name of the cache file."""
        raise NotImplementedError(
            "Database must define cache_file_prefix_name!")

    @property
    @abstractmethod
    def database_version(self) -> str:
        """Get the version of the database."""
        raise NotImplementedError("Database must define database_version!")

    @property
    @abstractmethod
    def database_root_path(self) -> Path:
        """Get the root path of the database."""
        raise NotImplementedError("Database must define database_root_path!")

    @property
    @abstractmethod
    def scenario_root_path(self) -> Path:
        """Get the root path of the scenario files."""
        raise NotImplementedError("Database must define scenario_root_path!")

    @property
    @abstractmethod
    def main_database(self) -> str:
        """Get the main database/scenario group name."""
        raise NotImplementedError("Database must define main_database!")

    @property
    @abstractmethod
    def scenarios(self) -> MappingProxyType[str, Scenarios]:
        """Get the scenarios for each scenario group."""
        raise NotImplementedError("Database must define scenarios!")

    @property
    @abstractmethod
    def cache_path(self) -> Path:
        """Get the cache path of the database."""
        raise NotImplementedError("Database must define cache_path!")

    @property
    @abstractmethod
    def num_workers(self) -> int:
        """Get the number of workers to use for processing the database."""
        raise NotImplementedError("Database must define num_workers!")

    @abstractmethod
    def get_main_database_scenario_data(self) -> Scenarios:
        """Get the scenario data for the main database."""
        raise NotImplementedError(
            "Database must define get_main_database_scenario_data!")

    @abstractmethod
    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]:
        """Get all scenario data from all scenario groups and keep their order the same."""
        raise NotImplementedError(
            "Database must define get_unique_scenario_data!")

    @abstractmethod
    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """Load scenario records from the database."""
        raise NotImplementedError(
            "Database must define load_scenario_records!")

    @abstractmethod
    def process_scenario_records(self) -> Sequence[DatasetRecord]:
        """Process scenario records from the database."""
        raise NotImplementedError(
            "Subclasses must define process_scenario_records method!")
