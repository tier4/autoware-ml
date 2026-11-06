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

"""Scenarios of the Argoverse 2 Motion Forecasting dataset.

The dataset ships one directory per scenario under a split directory, ``<split>/<scenario_id>/``,
and the split directory names are the official train / val / test split. The scenarios of a
split are therefore discovered from the corpus instead of being listed in a yaml file, as the
nuScenes scenarios are. A split holds up to 200k scenarios, so the content representations
stay compact: a digest of the scenario ids instead of a listing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import model_validator

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData, Scenarios
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)

# The split directory names of the dataset and the split each one serves.
_SPLIT_DIRECTORY_NAMES: Mapping[str, SplitType] = {
    "train": SplitType.TRAIN,
    "val": SplitType.VAL,
    "test": SplitType.TEST,
}


class AV2MotionScenarios(Scenarios):
    """
    Scenarios of the Argoverse 2 Motion Forecasting dataset, discovered from its split
    directories.

    Every ``dataset_params`` entry names one split directory under ``scenario_root_path`` with
    its ``dataset_name``: ``train``, ``val`` or ``test``. The subdirectories of the split
    directory are the scenarios, taken in sorted order. One scenario is one sample, so
    ``sample_steps`` is a stride over the scenarios of the split rather than over the frames of
    a scenario: ``sample_steps=10`` keeps every tenth scenario.
    """

    @model_validator(mode="after")
    def build_scenarios(self) -> AV2MotionScenarios:
        """
        Build the scenarios of every split named in ``dataset_params``.

        Returns:
          AV2MotionScenarios: AV2MotionScenarios class instance.
        """

        scenario_data: dict[SplitType, Sequence[ScenarioData]] = {}
        for dataset_params in self.dataset_params:
            split = self._split_of(dataset_params)
            if split in scenario_data:
                raise ValueError(
                    f"The split {dataset_params.dataset_name!r} is listed more than once in "
                    "dataset_params."
                )
            scenario_data[split] = self._build_scenario_data_for_split(dataset_params, split)
            logger.info(
                f"Discovered {len(scenario_data[split])} scenarios for split {split} under "
                f"{self.scenario_root_path / dataset_params.dataset_name}"
            )

        object.__setattr__(self, "scenario_data", scenario_data)
        return self

    @staticmethod
    def _split_of(dataset_params: DatasetParams) -> SplitType:
        """
        Resolve the split a dataset parameters entry names.

        Args:
          dataset_params: Dataset parameters whose ``dataset_name`` is a split directory name.

        Returns:
          SplitType: The split of the directory.
        """

        split = _SPLIT_DIRECTORY_NAMES.get(dataset_params.dataset_name)
        if split is None:
            raise ValueError(
                f"Unknown Argoverse 2 motion forecasting split {dataset_params.dataset_name!r}, "
                f"expected one of {sorted(_SPLIT_DIRECTORY_NAMES)}."
            )
        return split

    def _build_scenario_data_for_split(
        self, dataset_params: DatasetParams, split: SplitType
    ) -> Sequence[ScenarioData]:
        """
        Discover the scenarios of one split directory.

        Args:
          dataset_params: Dataset parameters naming the split directory, with the stride over
            its scenarios.
          split: The split of the directory.

        Returns:
          Sequence[ScenarioData]: One entry per kept scenario, in sorted id order.
        """

        split_path = self.scenario_root_path / dataset_params.dataset_name
        if not split_path.is_dir():
            raise ValueError(
                f"The split directory {split_path} of the Argoverse 2 motion forecasting "
                "dataset does not exist."
            )
        scenario_ids = self._list_scenario_ids(split_path)
        return [
            ScenarioData(
                dataset_params=dataset_params,
                scenario_id=scenario_id,
                scenario_version=dataset_params.dataset_name,
                vehicle_type=None,
                location=None,
            )
            for scenario_id in scenario_ids[:: dataset_params.sample_steps]
        ]

    @staticmethod
    def _list_scenario_ids(split_path: Path) -> list[str]:
        """
        List the scenario directories of a split.

        ``os.scandir`` takes the type of every entry from the directory listing, without a
        ``stat`` per scenario directory.

        Args:
          split_path: The split directory.

        Returns:
          list[str]: The scenario ids, which are the directory names, sorted.
        """

        with os.scandir(split_path) as entries:
            return sorted(entry.name for entry in entries if entry.is_dir())

    def __str__(self) -> str:
        """
        String representation of the scenarios, with the scenario count of every split instead
        of a listing of up to 200k scenarios.

        Returns:
          str: String representation of the scenarios.
        """

        dataset_params = ", ".join(str(dataset_param) for dataset_param in self.dataset_params)
        scenario_data = ", ".join(
            f"{split}: n={len(split_data)}" for split, split_data in self.scenario_data.items()
        )
        return (
            f"{self.__class__.__name__}(scenario_root_path={self.scenario_root_path}, "
            f"dataset_params=({dataset_params}), scenario_data=({scenario_data}))"
        )

    @property
    def hash_repr(self) -> str:
        """
        Get the representation of the scenarios that identifies their content.

        Every scenario of a split shares its dataset parameters and version, so the scenarios
        are identified by their count and a digest of their ids rather than a listing. As in the
        base class, ``scenario_root_path`` is left out so nodes with the data at different paths
        share a cache.

        Returns:
          str: Content representation of the scenarios.
        """

        dataset_params = ", ".join(dataset_param.hash_repr for dataset_param in self.dataset_params)
        scenario_data = ", ".join(
            f"{split}: n={len(split_data)} sha256={self._digest(split_data)}"
            for split, split_data in self.scenario_data.items()
        )
        return (
            f"{self.__class__.__name__}(dataset_params=({dataset_params}), "
            f"scenario_data=({scenario_data}))"
        )

    @staticmethod
    def _digest(split_data: Sequence[ScenarioData]) -> str:
        """
        Digest the ids of the scenarios of a split.

        Args:
          split_data: Scenarios of one split.

        Returns:
          str: SHA256 hex digest over the scenario ids, in order.
        """

        joined = "\n".join(scenario.scenario_id for scenario in split_data)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
