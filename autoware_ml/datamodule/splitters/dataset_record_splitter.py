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

from collections import defaultdict
from typing import Sequence
from types import MappingProxyType

import polars as pl

from autoware_ml.databases.scenarios import Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.types.dataset import SplitType


class ScenarioSplitter:
    """
    Common Splitter class to split a sequence of DatasetRecord into different splits
    (e.g. train, val, test) based on the given scenarios.
    """

    def __str__(self) -> str:
        """
        String representation of the splitter.

        Returns:
          str: String representation of the splitter.
        """
        return f"{self.__class__.__name__}()"

    def split_by_dataset_records(
        self,
        dataset_records: Sequence[DatasetRecord],
        scenarios: MappingProxyType[str, Scenarios],
    ) -> MappingProxyType[SplitType, Sequence[DatasetRecord]]:
        """
        Split the dataset records into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records: Sequence of dataset records to be split.
          scenarios: MappingProxyType[str, Scenarios] object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, Sequence[DatasetRecord]]: Mapping from split type to sequence of dataset records in that split.
        """
        splitter_scenarios = defaultdict(list)
        for scenario in scenarios.values():
            for split, scenario_data_list in scenario.scenario_data.items():
                # Convert scenario_ids to set for faster lookup
                unique_scenario_ids = set(
                    [scenario_data.scenario_id for scenario_data in scenario_data_list]
                )
                splitter_scenarios[split] += [
                    dataset_record
                    for dataset_record in dataset_records
                    if dataset_record.scenario_id in unique_scenario_ids
                ]

        return MappingProxyType(splitter_scenarios)

    def split_by_polars_dataframe(
        self,
        dataset_records_dataframe: pl.DataFrame,
        scenarios: MappingProxyType[str, Scenarios],
    ) -> MappingProxyType[SplitType, pl.DataFrame]:
        """
        Split the dataset dataframe into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records_dataframe: Polars DataFrame containing the dataset records to be split.
          scenarios: MappingProxyType[str, Scenarios] object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, pl.DataFrame]: Mapping from split type to Polars DataFrame of dataset records in that split.
        """
        splitter_scenarios = defaultdict(list)
        for scenario in scenarios.values():
            for split, scenario_data_list in scenario.scenario_data.items():
                unique_scenario_ids = set(
                    [scenario_data.scenario_id for scenario_data in scenario_data_list]
                )
                # Convert scenario_ids to set for faster lookup
                splitter_scenarios[split].append(
                    dataset_records_dataframe.filter(
                        pl.col(DatasetTableSchema.SCENARIO_ID.name).is_in(unique_scenario_ids)
                    )
                )

        return MappingProxyType(
            {split: pl.concat(dfs) for split, dfs in splitter_scenarios.items()}
        )
