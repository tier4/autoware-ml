from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Protocol
from types import MappingProxyType

from autoware_ml.databases.scenarios import Scenarios, ScenarioData
from autoware_ml.databases.schemas import DatasetRecord


class DatabaseInterface(Protocol):
    """Interface for database classes."""

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

    @abstractmethod
    def get_main_database_scenario_data(self) -> Scenarios:
        """Get the scenario data for the main database."""
        raise NotImplementedError("Database must define get_main_database_scenario_data!")

    @abstractmethod
    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]:
        """Get all scenario data from all scenario groups and keep their order the same."""
        raise NotImplementedError("Database must define get_unique_scenario_data!")

    @abstractmethod
    def load_scenario_records(self) -> Iterable[DatasetRecord]:
        """Load scenario records from the database."""
        raise NotImplementedError("Database must define load_scenario_records!")
