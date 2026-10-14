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
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.taxonomy import DatabaseTaxonomy
from autoware_ml.types.sensor import LidarChannel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class T4RecordsGeneratorWorkerParams:
    """
    Parameters for each scenario in T4Dataset to be
    processed by T4RecordsGenerator.

    Attributes:
      database_root_path: Root path of the T4 database.
      scenario_data: Scenario data.
      lidar_channel: Lidar channel each sample is keyed on.
      lidar_pointcloud_num_features: Number of features in the lidar pointcloud.
      box_annotation_dir: Directory of each scene with the category and instance tables to read
        box names from.
      taxonomy: Taxonomies the database labels are built with.
      box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
    """

    database_root_path: str
    scenario_data: ScenarioData
    lidar_channel: LidarChannel
    lidar_pointcloud_num_features: int
    box_annotation_dir: str
    taxonomy: DatabaseTaxonomy
    box3d_pipelines: Sequence[Box3DPipeline]


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
        lidar_channel=t4_records_generator_worker_params.lidar_channel,
        lidar_pointcloud_num_features=(
            t4_records_generator_worker_params.lidar_pointcloud_num_features
        ),
        box_annotation_dir=t4_records_generator_worker_params.box_annotation_dir,
        taxonomy=t4_records_generator_worker_params.taxonomy,
        box3d_pipelines=t4_records_generator_worker_params.box3d_pipelines,
    )
    # Generate DatasetRecords
    return t4_records_generator.generate_dataset_records()


class T4Dataset(BaseDatabase):
    """T4Dataset class."""

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: MappingProxyType[str, T4Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        taxonomy: DatabaseTaxonomy,
        lidar_channel: str,
        lidar_pointcloud_num_features: int,
        box3d_pipelines: Sequence[Box3DPipeline],
        lidar_intensity_scale: float,
        box_annotation_dir: str,
    ) -> None:
        """
        Initialize T4 dataset. Please refer to the BaseDatabase class for more details.

        Args:
          version: Version of the dataset.
          root_path: Root path where the actual annotation files are stored.
          scenarios: Scenario configurations for each scenario in {'scenario_group_name': scenario_config}.
          cache_path: Path to cache the dataset records.
          cache_file_prefix_name: Prefix name of the cache file, it will be <cache_file_prefix_name>_<dataset_hash>.parquet
          num_workers: Number of workers to use for processing the dataset.
          taxonomy: Taxonomies the database labels are built with.
          lidar_channel: Lidar channel each sample is keyed on.
          lidar_pointcloud_num_features: Number of features in the lidar pointcloud.
          box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
          lidar_intensity_scale: Intensity value of the strongest return in the stored point
            clouds.
          box_annotation_dir: Directory of each scene with the category and instance tables to
            read box names from.
        """

        logger.info("Initializing T4 dataset...")
        super().__init__(
            version=version,
            root_path=root_path,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
            taxonomy=taxonomy,
            box3d_pipelines=box3d_pipelines,
            lidar_intensity_scale=lidar_intensity_scale,
            lidar_pointcloud_num_features=lidar_pointcloud_num_features,
        )
        self._scenarios = scenarios
        self._lidar_channel = LidarChannel(lidar_channel)
        self._box_annotation_dir = box_annotation_dir

    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
          str: String representation of the database.
        """

        string = (
            f"T4Dataset(version={self._version}, "
            f"root_path={str(self._root_path)}, "
            f"cache path={str(self._cache_path)}, "
            f"cache file prefix name={self._cache_file_prefix_name}, "
            f"taxonomy={self._taxonomy}, "
            f"lidar_channel={self._lidar_channel.value}, "
            f"lidar_pointcloud_num_features={self._lidar_pointcloud_num_features}, "
            f"box_annotation_dir={self._box_annotation_dir}, "
            f"box3d_pipelines=[{', '.join([str(pipeline) for pipeline in self._box3d_pipelines])}], "
            f"{self.scenarios_string_repr}"
            f")"
        )
        return string

    @property
    def hash_repr(self) -> str:
        """
        Get the representation of the database that identifies the content of its cache.

        Extends the base representation with the lidar and annotation settings that shape the
        cached records.

        Returns:
          str: Content representation of the database.
        """

        return (
            f"{super().hash_repr}"
            f"(lidar_channel={self._lidar_channel.value}, "
            f"lidar_pointcloud_num_features={self._lidar_pointcloud_num_features}, "
            f"box_annotation_dir={self._box_annotation_dir})"
        )

    def __eq__(self, other: DatabaseInterface) -> bool:
        """
        Compare two databases by their version and scenario IDs.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """

        if not isinstance(other, T4Dataset):
            return False
        return str(self) == str(other)

    def process_scenario_records(self) -> None:
        """
        Process scenario records from the database.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        # Start the timer
        start_time = time.perf_counter()

        # 1) Get the polar schema and check if the caches exist
        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        df_hash = self.database_hash
        df_cache_path = self._cache_path / f"{self._cache_file_prefix_name}_{df_hash}.parquet"
        if df_cache_path.exists():
            logger.info(f"Cache file {df_cache_path} already exists, skip generating the caches")
            return

        # 1) Read all unique scenario data
        unique_scenario_data = self.get_unique_scenario_data()
        logger.info(
            f"Processing a total of {len(unique_scenario_data)} unique scenarios in T4Dataset"
        )

        # 2) Send the list to the multiprocessing or single processing the scenario
        # samples/frames
        scenario_sample_records = self._run_t4records_generator(unique_scenario_data)
        logger.info(f"Processed {len(scenario_sample_records)} scenario sample records")

        # 3) Save the scenario sample records to a polars .parquet file
        # Dump to a list of dictionaries to make it safer since it's using Pydantic.BaseModel
        scenario_dict_records = [record.to_dictionary() for record in scenario_sample_records]

        # 4) Get the polar schema
        polars_schema = self.get_polars_schema()
        logger.info(f"Parquet schema: {polars_schema}")

        # 5) Save the scenario sample records to a polars .parquet file
        df = pl.DataFrame(scenario_dict_records, schema=polars_schema)
        df.write_parquet(df_cache_path)
        logger.info(f"Saved the database cache to {df_cache_path} with the hash: {df_hash}")

        # End the timer
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(
            f"Elapsed time to process scenario records: {elapsed:.4f} seconds for the database: {self.version}"
        )

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
                database_root_path=str(self._root_path),
                scenario_data=scenario,
                lidar_channel=self._lidar_channel,
                lidar_pointcloud_num_features=self._lidar_pointcloud_num_features,
                box_annotation_dir=self._box_annotation_dir,
                taxonomy=self._taxonomy,
                box3d_pipelines=self._box3d_pipelines,
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
