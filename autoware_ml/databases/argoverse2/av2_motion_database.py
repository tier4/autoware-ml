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

"""Database of the Argoverse 2 Motion Forecasting dataset."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

import polars as pl
from tqdm import tqdm

from autoware_ml.databases.argoverse2.av2_motion_records_generator import (
    OBSERVED_STEPS,
    AV2MotionRecordsGenerator,
)
from autoware_ml.databases.argoverse2.av2_motion_scenarios import AV2MotionScenarios
from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.taxonomy import LabelVocabulary
from autoware_ml.types.dataset import DatasetFormat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AV2MotionRecordsGeneratorWorkerParams:
    """
    Parameters of one worker task: a chunk of scenarios, so a database of 250k scenarios does
    not submit 250k tasks.

    Attributes:
      database_root_path: Root path of the database.
      scenario_data: Scenarios of the chunk.
    """

    database_root_path: str
    scenario_data: tuple[ScenarioData, ...]


def _apply_av2_motion_records_generator(
    worker_params: AV2MotionRecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """
    Build the records of a chunk of scenarios in a worker.

    Args:
      worker_params: The chunk of scenarios.

    Returns:
      Sequence[DatasetRecord]: One record per scenario of the chunk, in order.
    """

    records: list[DatasetRecord] = []
    for scenario_data in worker_params.scenario_data:
        generator = AV2MotionRecordsGenerator(
            database_root_path=worker_params.database_root_path,
            scenario_data=scenario_data,
        )
        records.extend(generator.generate_dataset_records())
    return records


class AV2MotionDatabase(BaseDatabase):
    """
    Argoverse 2 Motion Forecasting database: one record per scenario pointing at the tracks
    and map files of the scenario, with no sensor data.
    """

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: Mapping[str, AV2MotionScenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        vocabulary: LabelVocabulary,
        scenarios_per_worker_task: int = 256,
    ) -> None:
        """
        Initialize the database. Please refer to the BaseDatabase class for more details.

        Args:
          version: Version of the database.
          root_path: Root path holding the split directories of the dataset.
          scenarios: Scenario configurations in {'scenario_group_name': scenarios}.
          cache_path: Path to cache the database records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<database_hash>.parquet
          num_workers: Number of workers to use for processing the database.
          vocabulary: Vocabulary of the object types of the tracks.
          scenarios_per_worker_task: Number of scenarios every worker task builds the records
            of.
        """

        logger.info("Initializing Argoverse 2 motion forecasting database...")
        super().__init__(
            version=version,
            root_path=root_path,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
            vocabulary=vocabulary,
        )
        if scenarios_per_worker_task < 1:
            raise ValueError(
                f"scenarios_per_worker_task must be positive, got {scenarios_per_worker_task}."
            )
        self._scenarios = scenarios
        self._scenarios_per_worker_task = scenarios_per_worker_task
        self._database_hash: str | None = None

    def __str__(self) -> str:
        """
        String representation of the database, with the scenario counts of every split rather
        than a listing of up to 250k scenarios.

        Returns:
          str: String representation of the database.
        """

        return (
            f"{self.__class__.__name__}(version={self._version}, "
            f"root_path={str(self._root_path)}, "
            f"cache path={str(self._cache_path)}, "
            f"cache file prefix name={self._cache_file_prefix_name}, "
            f"vocabulary={self._vocabulary}, "
            f"observed_steps={OBSERVED_STEPS}, "
            f"{self.scenarios_string_repr}"
            f")"
        )

    def __eq__(self, other: DatabaseInterface) -> bool:
        """
        Compare two databases by their version and scenario IDs.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """

        if not isinstance(other, AV2MotionDatabase):
            return False
        return str(self) == str(other)

    @property
    def hash_fields(self) -> Mapping[str, str]:
        """
        Get the explicit content fields that identify the cached records of the database.

        Extends the base fields with the observed steps that fix the current step of every
        record and the dataset format the records name. The root path stays out: the records
        store every file path relative to it. The vocabulary stays out too: the records store
        no labels, the tracks are resolved when a sample loads.

        Returns:
          Mapping[str, str]: Field name to its content representation, in a fixed order.
        """

        return {
            **super().hash_fields,
            "observed_steps": str(OBSERVED_STEPS),
            "dataset_format": DatasetFormat.AV2.value,
        }

    @property
    def database_hash(self) -> str:
        """
        Get a hash for the database based on the content of its cache, computed once: the
        scenarios are frozen and hashing them digests every scenario id.

        Returns:
          str: Hash of the database.
        """

        if self._database_hash is None:
            self._database_hash = super().database_hash
        return self._database_hash

    def process_scenario_records(self) -> None:
        """Process scenario records from the database."""

        start_time = time.perf_counter()

        df_hash = self.database_hash
        df_cache_path = self.cache_file_path
        if df_cache_path.exists():
            logger.info(f"Cache file {df_cache_path} already exists, skip generating the caches")
            return

        unique_scenario_data = self.get_unique_scenario_data()
        logger.info(
            f"Processing a total of {len(unique_scenario_data)} unique scenarios in "
            f"{self.__class__.__name__}"
        )

        records = self._run_records_generator(list(unique_scenario_data.values()))
        logger.info(f"Processed {len(records)} scenario records")

        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        df = pl.DataFrame([record.to_dictionary() for record in records], schema=polars_schema)
        df.write_parquet(df_cache_path)
        logger.info(f"Saved the database cache to {df_cache_path} with the hash: {df_hash}")

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Elapsed time to process scenario records: {elapsed:.4f} seconds for the database: "
            f"{self.version}"
        )

    def _run_records_generator(
        self, scenario_data: Sequence[ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Build the records of the scenarios, in chunks over the worker pool.

        Args:
          scenario_data: Scenarios to build the records of.

        Returns:
          Sequence[DatasetRecord]: One record per scenario, in the order of the scenarios.
        """

        chunk = self._scenarios_per_worker_task
        worker_params = [
            AV2MotionRecordsGeneratorWorkerParams(
                database_root_path=str(self._root_path),
                scenario_data=tuple(scenario_data[start : start + chunk]),
            )
            for start in range(0, len(scenario_data), chunk)
        ]

        records: list[DatasetRecord] = []
        if self._num_workers > 1:
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                futures = executor.map(_apply_av2_motion_records_generator, worker_params)
                for result in tqdm(futures, total=len(worker_params)):
                    records.extend(result)
        else:
            for worker_param in tqdm(worker_params, total=len(worker_params)):
                records.extend(_apply_av2_motion_records_generator(worker_param))
        return records
