from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import time
from typing import Sequence
from types import MappingProxyType

import polars as pl
from tqdm import tqdm

from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas import DatasetRecord
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class T4RecordsGeneratorWorkerParams:
    """
    Parameters for each scenario in T4Dataset to be
    processed by T4RecordsGenerator.

    Attributes:
      database_root_path: Root path of the T4 database.
      dataset_version: Version of the dataset.
      scenario_data: Scenario data.
    """

    database_root_path: str
    scenario_data: ScenarioData


def _apply_t4_records_generator(
    t4_records_generator_worker_params: T4RecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """
    Submit T4 records generator to the worker pool for a worker to process.

    Args:
      t4_records_generator_worker_params: T4 records generator worker parameters.
    Returns:
      Sequence[DatasetRecord]: Sequence of dataset records.
    """

    # Construct T4 records generator
    t4_records_generator = T4RecordsGenerator(
        database_root_path=t4_records_generator_worker_params.database_root_path,
        scenario_data=t4_records_generator_worker_params.scenario_data,
        sample_steps=t4_records_generator_worker_params.scenario_data.sample_steps,
        max_sweeps=t4_records_generator_worker_params.scenario_data.max_sweeps,
    )
    # Generate DatasetRecords
    return t4_records_generator.generate_dataset_records()


class T4Database(BaseDatabase):
    """T4Database class."""

    def __init__(
        self,
        database_version: str,
        database_root_path: str,
        scenarios: MappingProxyType[str, T4Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
    ) -> None:
        """
        Initialize T4 database. Please refer to the BaseDatabase class for more details.

        Args:
          database_version: Version of the database.
          database_root_path: Root path where the actual annotation files are stored.
          scenarios: Scenario configurations for each scenario in {'scenario_group_name': scenario_config}.
          cache_path: Path to cache the database records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<database_hash>.parquet
          num_workers: Number of workers to use for processing the database.
        """

        logger.info("Initializing T4 database...")
        super().__init__(
            database_version=database_version,
            database_root_path=database_root_path,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
        )
        self._scenarios = scenarios

    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
          str: String representation of the database.
        """

        string = (
            f"T4Database(database_version={self._database_version}, "
            f"database_root_path={str(self._database_root_path)}, "
            f"cache path={str(self._cache_path)}, "
            f"cache file prefix name={self._cache_file_prefix_name}, "
            f"{self.scenarios_string_repr}"
            f")"
        )
        return string

    def __eq__(self, other: DatabaseInterface) -> bool:
        """
        Compare two databases by their version and scenario IDs.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """

        if not isinstance(other, T4Database):
            return False
        return str(self) == str(other)

    def process_scenario_records(self) -> Sequence[DatasetRecord]:
        """
        Process scenario records from the database.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        # Start the timer
        start_time = time.perf_counter()

        # TODO (KokSeang): Read the cache if it exists, and return the records

        # First, read all unique scenario data
        unique_scenario_data = self.get_unique_scenario_data()
        logger.info(
            f"Processing a total of {len(unique_scenario_data)} unique scenarios in T4Database"
        )

        # Second, send the list to the multiprocessing or single processing the scenario
        # samples/frames
        scenario_sample_records = self._run_t4records_generator(unique_scenario_data)
        logger.info(f"Processed {len(scenario_sample_records)} scenario sample records")

        # Third, get the polar schema
        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        # Fourth, save the scenario sample records to a polars .parquet file
        # Dump to a list of dictionaries to make it safer since it's using Pydantic.BaseModel
        scenario_sample_records = [record.model_dump() for record in scenario_sample_records]
        df = pl.DataFrame(scenario_sample_records, schema=polars_schema)
        df_hash = hashlib.sha256(str(self).encode("utf-8")).hexdigest()
        df_cache_path = self._cache_path / f"{self._cache_file_prefix_name}_{df_hash}.parquet"
        df.write_parquet(df_cache_path)
        logger.info(f"Saved the database cache to {df_cache_path} with the hash: {df_hash}")

        # End the timer
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(
            f"Elapsed time to process scenario records: {elapsed:.4f} seconds for the database: {self.database_version}"
        )
        return scenario_sample_records

    def _run_t4records_generator(
        self, scenario_data: MappingProxyType[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Multi-process scenario records from the database.

        Args:
          scenario_data: Dict of Scenario ID to ScenarioData.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        # Group params for each worker
        worker_params = [
            T4RecordsGeneratorWorkerParams(
                database_root_path=self._database_root_path,
                scenario_data=scenario,
            )
            for scenario in scenario_data.values()
        ]

        flatten_records = []
        if self._num_workers > 1:
            # Run T4 records generator in multi processors
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                futures = executor.map(_apply_t4_records_generator, worker_params)
                for result in tqdm(futures, total=len(worker_params)):
                    flatten_records.extend(result)
                return flatten_records
        else:
            # Run T4 records generator in a single processor
            for worker_param in tqdm(worker_params, total=len(worker_params)):
                flatten_records.extend(_apply_t4_records_generator(worker_param))
            return flatten_records

    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """
        Load scenario records from the database.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        # TODO (KokSeang): Read the cache if it exists, and return the records
        raise NotImplementedError("Subclasses must implement load_scenario_records method!")
