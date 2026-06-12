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

from abc import abstractmethod
from collections import defaultdict
from typing import Sequence
from types import MappingProxyType


from autoware_ml.databases.scenarios import Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
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

    @abstractmethod
    def __call__(
        self,
        dataset_records: Sequence[DatasetRecord],
        scenarios: Scenarios,
    ) -> MappingProxyType[SplitType, Sequence[DatasetRecord]]:
        """
        Split the dataset records into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records: Sequence of dataset records to be split.
          scenarios: Scenarios object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, Sequence[DatasetRecord]]: Mapping from split type to sequence of dataset records in that split.
        """
        splitter_scenarios = defaultdict(list)
        scenario_data = scenarios.scenario_data
        for split, scenario_ids in scenario_data.items():
            # Convert scenario_ids to set for faster lookup
            unique_scenario_ids = set(scenario_ids)
            splitter_scenarios[split] = [
                dataset_record
                for dataset_record in dataset_records
                if dataset_record.scenario_id in unique_scenario_ids
            ]
        return splitter_scenarios
