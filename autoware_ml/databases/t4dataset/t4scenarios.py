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

from typing import Sequence
import logging
from collections import defaultdict
import yaml
from types import MappingProxyType

from pydantic import model_validator

from autoware_ml.types.dataset import SplitType
from autoware_ml.databases.scenarios import ScenarioData, Scenarios, DatasetParams

logger = logging.getLogger(__name__)

_TRAFFIC_CONE_BARRIER_BBOX_STATUS_TEXTS: MappingProxyType[str, bool | None] = MappingProxyType(
    {"true": True, "false": False, "none": None}
)


def _parse_traffic_cone_barrier_bbox_status(status_text: str) -> bool | None:
    """
    Parse the traffic cone / barrier annotation status segment of a scenario ID.

    Args:
      status_text: The last segment of a fully qualified scenario ID, ``true`` when the cones
        and barriers of the scenario are annotated, ``false`` when they are not, ``none`` when
        it is unknown. Case-insensitive.

    Returns:
      bool | None: The annotation status, None when unknown.

    Raises:
      ValueError: If the text is none of the accepted values. ``bool(text)`` is never used,
        since it would read ``false`` as True.
    """

    try:
        return _TRAFFIC_CONE_BARRIER_BBOX_STATUS_TEXTS[status_text.lower()]
    except KeyError:
        raise ValueError(
            f"Invalid traffic cone / barrier annotation status {status_text!r}, expected one of "
            f"{sorted(_TRAFFIC_CONE_BARRIER_BBOX_STATUS_TEXTS)}."
        ) from None


class T4Scenarios(Scenarios):
    """
    T4Scenarios class inherits from Scenarios and defines the logic for building
    scenario data for T4Dataset.
    """

    @model_validator(mode="after")
    def build_scenarios(self) -> T4Scenarios:
        """
        Build scenarios from database scenarios, and
        overwrite the scenario_data attribute.

        Returns:
          T4Scenarios: T4Scenarios class instance.
        """

        scenario_data = defaultdict(list)
        for dataset_param in self.dataset_params:
            db_yaml_path = self.scenario_root_path / (dataset_param.dataset_name + ".yaml")
            logger.info(f"Loading database scenarios from {db_yaml_path}")
            with open(db_yaml_path, "r") as f:
                db_scenarios: MappingProxyType[str, Sequence[str]] = yaml.safe_load(f)

            scenario_splits = self._build_scenario_splits(db_scenarios, dataset_param)
            for split, scenarios in scenario_splits.items():
                scenario_data[split] += scenarios

        object.__setattr__(self, "scenario_data", scenario_data)
        for split, scenarios in scenario_data.items():
            logger.info(f"Loaded total of {len(scenarios)} scenarios for split {split}")
        return self

    @staticmethod
    def _build_scenario_data(scenario_id: str, dataset_params: DatasetParams) -> ScenarioData:
        """
        Build scenario data from a scenario ID, either ``<scene>/<version>`` or
        ``<scene>/<version>/<location>/<vehicle_type>/<cone_barrier_status>`` where the last
        segment is ``true``, ``false`` or ``none`` (annotation status unknown).

        Args:
          scenario_id: Scenario ID.
          dataset_params: Dataset parameters.

        Returns:
          ScenarioData: Scenario data.
        """

        dataset_scene_info = scenario_id.split("/")
        if len(dataset_scene_info) == 5:
            scenario_id, version, city, vehicle_type, status_text = dataset_scene_info
            traffic_cone_barrier_bbox_status = _parse_traffic_cone_barrier_bbox_status(status_text)
        elif len(dataset_scene_info) == 2:
            scenario_id, version = dataset_scene_info
            city = vehicle_type = traffic_cone_barrier_bbox_status = None
        else:
            raise ValueError(f"Invalid scenario ID: {scenario_id}")

        return ScenarioData(
            dataset_params=dataset_params,
            scenario_id=scenario_id,
            scenario_version=version,
            vehicle_type=vehicle_type,
            location=city,
            traffic_cone_barrier_bbox_status=traffic_cone_barrier_bbox_status,
        )

    def _build_scenario_splits(
        self, db_scenarios: MappingProxyType[str, Sequence[str]], dataset_params: DatasetParams
    ) -> MappingProxyType[SplitType, Sequence[ScenarioData]]:
        """
        Build splits from a database scenarios.

        Args:
          db_scenarios: Dictionary of split type to a list of scenario IDs.
          dataset_param: Dataset parameters.

        Returns:
          MappingProxyType[SplitType, Sequence[ScenarioData]]: Dictionary of SplitType to
          a list of ScenarioData for the corresponding split.
        """

        scenario_splits = {}
        for split in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST]:
            selected_scenarios = db_scenarios.get(split, [])
            scenario_splits[split] = [
                self._build_scenario_data(scenario_id=scenario_id, dataset_params=dataset_params)
                for scenario_id in selected_scenarios
            ]
        return MappingProxyType(scenario_splits)
