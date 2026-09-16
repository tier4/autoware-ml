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

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import time
from typing import Sequence
from types import MappingProxyType

import polars as pl
from tqdm import tqdm

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.nuscenes.nuscenes_records_generator import NuScenesRecordsGenerator
from autoware_ml.databases.nuscenes.nuscenes_scenarios import NuScenesScenarios
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NuScenesRecordsGeneratorWorkerParams:
    """
    Parameters for each scenario in NuScenesDataset to be
    processed by NuScenesRecordsGenerator.

    Attributes:
      database_root_path: Root path of the NuScenes database.
      scenario_data: Scenario data.
      lidar_pointcloud_num_features: Number of features in the lidar pointcloud.
    """

    database_root_path: str
    scenario_data: ScenarioData
    lidar_pointcloud_num_features: int
    ignore_label_index: int
    box3d_pipelines: Sequence[Box3DPipeline]


def _apply_nuscenes_records_generator(
    worker_params: NuScenesRecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """
    Submit NuScenes records generator to the worker pool for a worker to process.

    Args:
      worker_params: NuScenes records generator worker parameters.
    Returns:
      Sequence[DatasetRecord]: Sequence of dataset records.
    """

    nuscenes_records_generator = NuScenesRecordsGenerator(
        database_root_path=worker_params.database_root_path,
        scenario_data=worker_params.scenario_data,
        sample_steps=worker_params.scenario_data.sample_steps,
        max_sweeps=worker_params.scenario_data.max_sweeps,
        lidar_pointcloud_num_features=worker_params.lidar_pointcloud_num_features,
        ignore_label_index=worker_params.ignore_label_index,
        box3d_pipelines=worker_params.box3d_pipelines,
    )
    return nuscenes_records_generator.generate_dataset_records()


class NuScenesDataset(BaseDatabase):
    """NuScenesDataset class."""

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: MappingProxyType[str, NuScenesScenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        class_names: Sequence[str],
        ignore_label_index: int,
        label_remapper: MappingProxyType[str, str] | None,
        lidar_pointcloud_num_features: int,
        box3d_pipelines: Sequence[Box3DPipeline],
    ) -> None:
        """
        Initialize NuScenes dataset. Please refer to the BaseDatabase class for more details.

        Args:
          version: Version of the dataset.
          root_path: Root path where the NuScenes version directories are stored.
          scenarios: Scenario configurations for each scenario in {'scenario_group_name': scenario_config}.
          cache_path: Path to cache the dataset records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<dataset_hash>.parquet
          num_workers: Number of workers to use for processing the dataset.
          class_names: List of class names in the dataset, used for category mapping.
          ignore_label_index: Index to use for ignored labels.
          label_remapper: Mapping to remap label names, if needed.
          lidar_pointcloud_num_features: Number of features in the lidar pointcloud.
          box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
        """

        logger.info("Initializing NuScenes dataset...")
        super().__init__(
            version=version,
            root_path=root_path,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
            class_names=class_names,
            label_remapper=label_remapper,
            box3d_pipelines=box3d_pipelines,
            ignore_label_index=ignore_label_index,
        )
        self._scenarios = scenarios
        self._lidar_pointcloud_num_features = lidar_pointcloud_num_features

    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
          str: String representation of the database.
        """

        string = (
            f"NuScenesDataset(version={self._version}, "
            f"root_path={str(self._root_path)}, "
            f"cache path={str(self._cache_path)}, "
            f"cache file prefix name={self._cache_file_prefix_name}, "
            f"class_names={self._class_names}, "
            f"label_remapper={self._label_remapper}, "
            f"ignore_label_index={self._ignore_label_index}, "
            f"box3d_pipelines=[{', '.join([str(pipeline) for pipeline in self._box3d_pipelines])}], "
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

        if not isinstance(other, NuScenesDataset):
            return False
        return str(self) == str(other)

    def process_scenario_records(self) -> None:
        """
        Process scenario records from the database.
        """

        start_time = time.perf_counter()

        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        df_hash = self.database_hash
        df_cache_path = self._cache_path / f"{self._cache_file_prefix_name}_{df_hash}.parquet"
        if df_cache_path.exists():
            logger.info(f"Cache file {df_cache_path} already exists, skip generating the caches")
            return

        unique_scenario_data = self.get_unique_scenario_data()
        logger.info(
            f"Processing a total of {len(unique_scenario_data)} unique scenarios in NuScenesDataset"
        )

        scenario_sample_records = self._run_nuscenes_records_generator(unique_scenario_data)
        logger.info(f"Processed {len(scenario_sample_records)} scenario sample records")

        scenario_dict_records = [record.to_dictionary() for record in scenario_sample_records]

        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        df = pl.DataFrame(scenario_dict_records, schema=polars_schema)
        df.write_parquet(df_cache_path)
        logger.info(f"Saved the database cache to {df_cache_path} with the hash: {df_hash}")

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(
            f"Elapsed time to process scenario records: {elapsed:.4f} seconds for the database: {self.version}"
        )

    def _run_nuscenes_records_generator(
        self, scenario_data: MappingProxyType[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Multi-process scenario records from the database.

        Args:
          scenario_data: Dict of Scenario ID to ScenarioData.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        worker_params = [
            NuScenesRecordsGeneratorWorkerParams(
                database_root_path=str(self._root_path),
                scenario_data=scenario,
                lidar_pointcloud_num_features=self._lidar_pointcloud_num_features,
                ignore_label_index=self._ignore_label_index,
                box3d_pipelines=self._box3d_pipelines,
            )
            for scenario in scenario_data.values()
        ]

        flatten_records = []
        if self._num_workers > 1:
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                futures = executor.map(_apply_nuscenes_records_generator, worker_params)
                for result in tqdm(futures, total=len(worker_params)):
                    flatten_records.extend(result)
                return flatten_records
        else:
            for worker_param in tqdm(worker_params, total=len(worker_params)):
                flatten_records.extend(_apply_nuscenes_records_generator(worker_param))
            return flatten_records
