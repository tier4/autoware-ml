from abc import abstractmethod
from pathlib import Path
from typing import Protocol, ImmutableMapping, Iterable

from autoware_ml.common.enums import TaskType
from autoware_ml.databases.scenarios import Scenarios


class DatabaseInterface(Protocol):
    """Interface for database classes."""

    @property
    @abstractmethod
    def datatabase_version(self) -> str:
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
    def scenarios(self) -> ImmutableMapping[str, Scenarios]:
        """Get the scenarios for each scenario group."""
        raise NotImplementedError("Database must define scenarios!")

    @property
    @abstractmethod
    def task_types(self) -> Iterable[TaskType]:
        """Get the task types supported by the database."""
        raise NotImplementedError("Database must define task_types!")
