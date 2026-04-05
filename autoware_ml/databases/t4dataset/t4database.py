import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import time
from typing import Sequence
from types import MappingProxyType

import polars as pl
from tqdm import tqdm

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas import DatasetRecord
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class T4RecordsGeneratorWorkerParams:
    database_root_path: str
    scenario_data: ScenarioData


def _apply_t4_records_generator(
    t4_records_generator_worker_params: T4RecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """Submit T4 records generator to the worker pool for a worker to process."""

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
    """T4 database class."""

    def __init__(
        self,
        database_version: str,
        database_root_path: str,
        scenario_root_path: str,
        # scenario_configs: MappingProxyType[str, DictConfig],
        scenarios: MappingProxyType[str, T4Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        main_database: str,
        num_workers: int = 16,
    ) -> None:
        """Initialize T4 database. Please refer to the BaseDatabase class for more details."""
        logger.info("Initializing T4 database...")
        super().__init__(
            database_version=database_version,
            database_root_path=database_root_path,
            scenario_root_path=scenario_root_path,
            scenarios=scenarios,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            main_database=main_database,
        )
        self.num_workers = num_workers

    def process_scenario_records(self) -> Sequence[DatasetRecord]:
        """Load scenario records from the database."""

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
        if self.num_workers > 1:
            scenario_sample_records = self._multi_process_scenario_records(unique_scenario_data)
        else:
            scenario_sample_records = self._single_process_scenario_records(unique_scenario_data)
        logger.info(f"Processed {len(scenario_sample_records)} scenario sample records")

        # Third, get the polar schema
        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        # Fourth, save the scenario sample records to a polars .parquet file
        df = pl.DataFrame(scenario_sample_records, schema=polars_schema)
        df_hash = hashlib.sha256(str(self).encode("utf-8")).hexdigest()
        df_cache_path = self._cache_path / f"{self.cache_file_prefix_name}_{df_hash}.parquet"
        df.write_parquet(df_cache_path)
        logger.info(f"Saved the database cache to {df_cache_path} with the hash: {df_hash}")

        # End the timer
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(
            f"Elapsed time to process scenario records: {elapsed:.4f} seconds for the database: {self.database_version}"
        )
        return scenario_sample_records

    def _multi_process_scenario_records(
        self, scenario_data: MappingProxyType[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """Multi-process scenario records from the database."""
        # Group params for each worker
        worker_params = [
            T4RecordsGeneratorWorkerParams(
                database_root_path=self.database_root_path,
                scenario_data=scenario,
            )
            for scenario in scenario_data.values()
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = executor.map(_apply_t4_records_generator, worker_params)
            flatten_records = []
            for result in tqdm(futures, total=len(worker_params)):
                flatten_records.extend(result)

            return flatten_records

    def _single_process_scenario_records(
        self, scenario_data: MappingProxyType[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """Single-process scenario records from the database."""
        flatten_records = []
        for scenario in tqdm(scenario_data.values(), total=len(scenario_data)):
            t4_records_generator = T4RecordsGenerator(
                database_root_path=self.database_root_path,
                scenario_data=scenario,
                sample_steps=scenario.sample_steps,
                max_sweeps=scenario.max_sweeps,
            )
            flatten_records.extend(t4_records_generator.generate_dataset_records())
        return flatten_records

    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """Load scenario records from the database."""
        # TODO (KokSeang): Read the cache if it exists, and return the records
        raise NotImplementedError("Subclasses must implement load_scenario_records method!")
