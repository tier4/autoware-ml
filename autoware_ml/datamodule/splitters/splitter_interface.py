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
from typing import Sequence, Protocol
from types import MappingProxyType

import polars as pl

from autoware_ml.databases.scenarios import Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.types.dataset import SplitType


class SplitterInterface(Protocol):
    """
    Protocol for splitter classes that defines the common interface for every dataset type.
    Each protocol class must support either
    split_by_dataset_records or split_by_polars_dataframe or both depending on the dataset type.
    """

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the splitter.

        Returns:
          str: String representation of the splitter.
        """

        raise NotImplementedError("Splitter must define __str__!")

    @abstractmethod
    def split_by_dataset_records(
        self,
        dataset_records: Sequence[DatasetRecord],
        scenarios: MappingProxyType[str, Scenarios],
    ) -> MappingProxyType[SplitType, Sequence[DatasetRecord]]:
        """
        Split the dataset records into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records: Sequence of dataset records to be split.
          scenarios: Scenarios object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, Sequence[DatasetRecord]]: Mapping from split type to sequence of dataset records in that split.
        """
        raise NotImplementedError("Splitter must define split_by_dataset_records()!")

    @abstractmethod
    def split_by_polars_dataframe(
        self,
        dataset_records_dataframe: pl.DataFrame,
        scenarios: MappingProxyType[str, Scenarios],
    ) -> MappingProxyType[SplitType, pl.DataFrame]:
        """
        Split the dataset dataframe into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be split.
          scenarios: Scenarios object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, pl.DataFrame]: Mapping from split type to Polars DataFrame of dataset records in that split.
        """
        raise NotImplementedError("Splitter must define split_by_polars_dataframe()!")
