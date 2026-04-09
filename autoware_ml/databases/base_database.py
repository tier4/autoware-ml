from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence
from types import MappingProxyType

import polars as pl

from autoware_ml.databases.scenarios import Scenarios, ScenarioData
from autoware_ml.databases.schemas import DatasetRecord, DatasetTableSchema

logger = logging.getLogger(__name__)


class BaseDatabase:
    """
    Interface for database classes.
    """

    def __init__(
        self,
        database_version: str,
        database_root_path: str,
        scenario_root_path: str,
        scenarios: MappingProxyType[str, Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
    ) -> None:
        """
        Initialize database interface.
        Args:
          database_version: Version of the database.
          database_root_path: Root path where the actual annotation files are stored.
          scenario_root_path: Root path where the scenario yaml files are stored.
          scenarios: Scenario configurations for each scenario in {'scenario_group_name': scenario_config}.
          cache_path: Path to cache the database records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<database_hash>.parquet
          num_workers: Number of workers to use for processing the database.
        """
        self._database_version = database_version
        self._database_root_path = Path(database_root_path)
        self._scenario_root_path = Path(scenario_root_path)
        self._cache_path = Path(cache_path)
        self._cache_file_prefix_name = cache_file_prefix_name
        self._scenarios: MappingProxyType[str, Scenarios] = scenarios
        self._num_workers = num_workers

        # Create cache output path if it doesn't exist
        self._cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Database initialized with version: {self.database_version}, root path: {self.database_root_path}, scenario root path: {self.scenario_root_path}, cache path: {self.cache_path}, cache file prefix name: {self.cache_file_prefix_name}"
        )

    def __str__(self) -> str:
        """String representation of the database."""
        string = f"BaseDatabase(database_version={self.database_version}, database_root_path={str(self.database_root_path)}, scenario_root_path={str(self.scenario_root_path)}, cache path={str(self.cache_path)}, cache file prefix name={self.cache_file_prefix_name}"
        string += ", scenarios=("
        for scenario_group, scenarios in self.scenarios.items():
            string += f"{scenario_group}: {scenarios}, "
        string += "))"
        return string

    def __eq__(self, other: BaseDatabase) -> bool:
        """Compare two databases by their version and scenario IDs."""
        return (
            self.database_version == other.database_version
            and self.database_root_path == other.database_root_path
            and self.scenario_root_path == other.scenario_root_path
            and self.main_database == other.main_database
            and self.scenarios == other.scenarios
        )

    def __hash__(self) -> int:
        """Hash the database by its version and scenario IDs."""
        return hash(str(self))

    @property
    def database_version(self) -> str:
        """Get the version of the database."""
        return self._database_version

    @property
    def database_root_path(self) -> Path:
        """Get the root path of the database."""
        return self._database_root_path

    @property
    def scenario_root_path(self) -> Path:
        """Get the root path of the scenario files."""
        return self._scenario_root_path

    @property
    def main_database(self) -> str:
        """Get the main database/scenario group name."""
        return self._main_database

    @property
    def scenarios(self) -> Mapping[str, Scenarios]:
        """Get the scenarios for each scenario group."""
        return self._scenarios

    @property
    def cache_path(self) -> Path:
        """Get the cache path of the database."""
        return self._cache_path

    @property
    def cache_file_prefix_name(self) -> str:
        """Get the prefix name of the cache file."""
        return self._cache_file_prefix_name

    @property
    def num_workers(self) -> int:
        """Get the number of workers to use for processing the database."""
        return self._num_workers

    def get_polars_schema(self) -> pl.Schema:
        """Get the polars schema for the database."""
        return DatasetTableSchema.to_polars_schema()

    def get_main_database_scenario_data(self) -> Scenarios:
        """Get the scenario data for the main database."""
        main_database_scenario_data = self.scenarios.get(self.main_database, None)
        if main_database_scenario_data is None:
            raise ValueError(f"Main database {self.main_database} not found!")
        return main_database_scenario_data

    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]:
        """Get all scenario data from all scenario groups and keep their order the same."""
        unique_scenarios = {}
        for _, scenarios in self.scenarios.items():
            for scenario in scenarios.get_all_scenario_data():
                if scenario.scenario_id not in unique_scenarios:
                    unique_scenarios[scenario.scenario_id] = scenario
        return unique_scenarios

    def process_scenario_records(self) -> Sequence[DatasetRecord]:
        """Process scenario records from the database."""
        raise NotImplementedError("Subclasses must implement process_scenario_records method!")
