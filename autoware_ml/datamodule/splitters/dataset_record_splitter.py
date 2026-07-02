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
from typing import Sequence, Mapping
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
        scenario_ids_by_split: Mapping[SplitType, set[str]] = defaultdict(set)
        for scenarios_obj in scenarios.values():
            for split, scenario_data_list in scenarios_obj.scenario_data.items():
                scenario_ids_by_split[split].update(sd.scenario_id for sd in scenario_data_list)

        records_by_scenario_id: Mapping[str, list[DatasetRecord]] = defaultdict(list)
        for record in dataset_records:
            records_by_scenario_id[record.scenario_id].append(record)

        split_records: Mapping[SplitType, Sequence[DatasetRecord]] = {}
        for split, scenario_ids in scenario_ids_by_split.items():
            records: Sequence[DatasetRecord] = []
            for scenario_id in scenario_ids:
                records.extend(records_by_scenario_id.get(scenario_id, []))
            split_records[split] = records

        return MappingProxyType(split_records)

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
        scenario_ids_by_split: Mapping[SplitType, set[str]] = defaultdict(set)
        for scenarios_obj in scenarios.values():
            for split, scenario_data_list in scenarios_obj.scenario_data.items():
                scenario_ids_by_split[split].update(sd.scenario_id for sd in scenario_data_list)

        split_dfs: Mapping[SplitType, pl.DataFrame] = {}
        for split, scenario_ids in scenario_ids_by_split.items():
            split_dfs[split] = dataset_records_dataframe.filter(
                pl.col(DatasetTableSchema.SCENARIO_ID.name).is_in(scenario_ids)
            )

        return MappingProxyType(split_dfs)
