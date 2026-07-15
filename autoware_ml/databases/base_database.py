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

import hashlib
import logging
from pathlib import Path
from typing import Mapping, Sequence
from types import MappingProxyType

import polars as pl

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.scenarios import Scenarios, ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema

logger = logging.getLogger(__name__)


class BaseDatabase:
    """Definition of a base database class that will be inherited by every dataset type."""

    def __init__(
        self,
        version: str,
        root_path: str,
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        class_names: Sequence[str],
        label_remapper: Mapping[str, str] | None,
        ignore_label_index: int,
        box3d_pipelines: Sequence[Box3DPipeline],
    ) -> None:
        """
        Initialize BaseDatabase.

        Args:
          version: Version of the database.
          root_path: Root path where the actual annotation files are stored.
          cache_path: Path to cache the database records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<database_hash>.parquet
          num_workers: Number of workers to use for processing the database.
          class_names: List of class names in the database, used for category mapping.
          label_remapper: Mapping to remap label names, if needed.
          ignore_label_index: Index to use for ignored labels.
          box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
        """

        self._version = version
        self._root_path = Path(root_path)
        self._cache_path = Path(cache_path)
        self._cache_file_prefix_name = cache_file_prefix_name
        self._num_workers = num_workers
        self._class_names = class_names
        self._label_remapper = label_remapper
        self._ignore_label_index = ignore_label_index
        self._box3d_pipelines = box3d_pipelines

        # Create cache output path if it doesn't exist
        self._cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Database initialized with version: {self._version}, "
            f"root path: {self._root_path}, "
            f"cache path: {self._cache_path}, "
            f"cache file prefix name: {self._cache_file_prefix_name}, "
            f"class names: {self._class_names}, "
            f"label remapper: {self._label_remapper}, "
            f"ignore label index: {self._ignore_label_index}, "
            f"box3d pipelines: [{', '.join([str(pipeline) for pipeline in self._box3d_pipelines])}]"
        )

        self._scenarios: MappingProxyType[str, Scenarios] = {}

    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
          str: String representation of the database.
        """

        raise NotImplementedError("Subclasses must implement __str__ method!")

    def __eq__(self, other: BaseDatabase) -> bool:
        """
        Compare two databases by their version and scenario IDs.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """

        raise NotImplementedError("Subclasses must implement __eq__ method!")

    def __hash__(self) -> int:
        """
        Hash the database by its version and scenario IDs.

        Returns:
          int: Hash of the database.
        """

        return hash(str(self))

    @property
    def class_names(self) -> Sequence[str]:
        """
        Get the class names in the database.

        Returns:
          Sequence[str]: Class names in the database.
        """

        return self._class_names

    @property
    def label_remapper(self) -> Mapping[str, str] | None:
        """
        Get the label remapper in the database.

        Returns:
          Mapping[str, str] | None: Label remapper in the database.
        """

        return self._label_remapper

    @property
    def ignore_label_index(self) -> int:
        """
        Get the ignore label index in the database.

        Returns:
          int: Ignore label index in the database.
        """

        return self._ignore_label_index

    @property
    def scenarios_string_repr(self) -> str:
        """
        Get string representation of the scenarios.

        Returns:
          str: String representation of the scenarios.
        """

        string = "scenarios=("
        for scenario_group, scenarios in self.scenarios.items():
            string += f"{scenario_group}: {scenarios}, "
        string += ")"
        return string

    @property
    def version(self) -> str:
        """
        Get the version of the database.

        Returns:
          str: Version of the database.
        """

        return self._version

    @property
    def scenarios(self) -> Mapping[str, Scenarios]:
        """
        Get the scenarios for each scenario group.

        Returns:
          Mapping[str, Scenarios]: Dictionary of scenario group name to scenarios.
        """

        return self._scenarios

    @property
    def database_hash(self) -> str:
        """
        Get a hash for the database based on its version and scenarios.

        Returns:
          str: Hash of the database.
        """
        hash_str = str(self)
        polars_schema = self.get_polars_schema()
        # Convert the polars schema to a string representation
        schema_str = str(polars_schema)
        hash_str += schema_str
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

    def get_polars_schema(self) -> pl.Schema:
        """
        Get the polars schema for the database.

        Returns:
          pl.Schema: Polars schema.
        """

        return DatasetTableSchema.to_polars_schema()

    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]:
        """
        Get all scenario data from all scenario groups and keep their order the same.

        Returns:
          MappingProxyType[str, ScenarioData]: Dictionary of scenario ID to scenario data.
        """

        unique_scenarios = {}
        for _, scenarios in self.scenarios.items():
            for scenario in scenarios.get_all_scenario_data():
                if scenario.scenario_id not in unique_scenarios:
                    unique_scenarios[scenario.scenario_id] = scenario
        return unique_scenarios

    def process_scenario_records(self) -> None:
        """Process scenario records from the database."""

        raise NotImplementedError("Subclasses must implement process_scenario_records method!")

    def load_polars_scenario_dataframe(self) -> pl.DataFrame:
        """
        Load the scenario records as a Polars dataframe.

        Returns:
          pl.DataFrame: Polars dataframe of the scenario records.
        """
        df_cache_path = (
            self._cache_path / f"{self._cache_file_prefix_name}_{self.database_hash}.parquet"
        )
        if not df_cache_path.exists():
            raise IOError(
                f"Cache file {df_cache_path} does not exist. "
                f"Please run process_scenario_records() to generate the cache file "
                f"before loading the scenario records."
            )

        df = pl.read_parquet(df_cache_path, schema=self.get_polars_schema())
        logger.info(f"Loaded scenario records as Polars dataframe from cache file {df_cache_path}")
        return df

    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """
        Load scenario records from the database.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """
        df = self.load_polars_scenario_dataframe()
        # Convert the dataframe to a list of dataset records
        dataset_records = [DatasetRecord.load_from_dictionary(record) for record in df.to_dicts()]
        logger.info(f"Loaded {len(dataset_records)} scenario records from database")
        return dataset_records
