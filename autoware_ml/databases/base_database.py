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
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
    ) -> None:
        """
        Initialize database interface.
        Args:
          database_version: Version of the database.
          database_root_path: Root path where the actual annotation files are stored.
          cache_path: Path to cache the database records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<database_hash>.parquet
          num_workers: Number of workers to use for processing the database.
        """
        self._database_version = database_version
        self._database_root_path = Path(database_root_path)
        self._cache_path = Path(cache_path)
        self._cache_file_prefix_name = cache_file_prefix_name
        self._num_workers = num_workers

        # Create cache output path if it doesn't exist
        self._cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Database initialized with version: {self._database_version}, "
            f"root path: {self._database_root_path}, "
            f"cache path: {self._cache_path}, "
            f"cache file prefix name: {self._cache_file_prefix_name}"
        )

        self._scenarios: MappingProxyType[str, Scenarios] = {}

    def __str__(self) -> str:
        """String representation of the database."""
        raise NotImplementedError("Subclasses must implement __str__ method!")

    def __eq__(self, other: BaseDatabase) -> bool:
        """Compare two databases by their version and scenario IDs."""
        raise NotImplementedError("Subclasses must implement __eq__ method!")

    def __hash__(self) -> int:
        """Hash the database by its version and scenario IDs."""
        return hash(str(self))

    @property
    def scenarios_string_repr(self) -> str:
        """Get string representation of the scenarios."""
        string = "scenarios=("
        for scenario_group, scenarios in self.scenarios.items():
            string += f"{scenario_group}: {scenarios}, "
        string += ")"
        return string

    @property
    def database_version(self) -> str:
        """Get the version of the database."""
        return self._database_version

    @property
    def scenarios(self) -> Mapping[str, Scenarios]:
        """Get the scenarios for each scenario group."""
        return self._scenarios

    def get_polars_schema(self) -> pl.Schema:
        """Get the polars schema for the database."""
        return DatasetTableSchema.to_polars_schema()

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
