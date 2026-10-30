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
from collections import defaultdict
from typing import Mapping, Sequence
from types import MappingProxyType

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_splits
from pydantic import model_validator

from autoware_ml.types.dataset import SplitType
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData, Scenarios

logger = logging.getLogger(__name__)

# Maps a NuScenes version directory name to the nuscenes-devkit `nuscenes.utils.splits` scene-name
# lists that apply to it. The labels of the official test scenes are not public, so the validation
# scenes also serve the test split.
_VERSION_TO_SPLIT_SCENE_NAMES = {
    "v1.0-trainval": {
        SplitType.TRAIN: nuscenes_splits.train,
        SplitType.VAL: nuscenes_splits.val,
        SplitType.TEST: nuscenes_splits.val,
    },
    "v1.0-mini": {
        SplitType.TRAIN: nuscenes_splits.mini_train,
        SplitType.VAL: nuscenes_splits.mini_val,
        SplitType.TEST: nuscenes_splits.mini_val,
    },
}


class NuScenesScenarios(Scenarios):
    """
    NuScenesScenarios class inherits from Scenarios and defines the logic for building scenario
    data for a NuScenesDataset.

    Train/val/test split membership is derived from nuscenes-devkit's own
    official scene-name split lists (`nuscenes.utils.splits`).
    """

    @model_validator(mode="after")
    def build_scenarios(self) -> NuScenesScenarios:
        """
        Build scenarios by loading the NuScenes version tables via nuscenes-devkit, and splitting
        scenes into train/val/test according to nuscenes-devkit's official scene-name split lists.

        Returns:
          NuScenesScenarios: NuScenesScenarios class instance.
        """

        scenario_data = defaultdict(list)
        for dataset_param in self.dataset_params:
            scenario_data_for_version = self._build_scenario_data_for_version(dataset_param)
            for split, scenarios in scenario_data_for_version.items():
                scenario_data[split] += scenarios

        object.__setattr__(self, "scenario_data", scenario_data)
        for split, scenarios in scenario_data.items():
            logger.info(f"Loaded total of {len(scenarios)} scenarios for split {split}")
        return self

    def _build_scenario_data_for_version(
        self, dataset_params: DatasetParams
    ) -> MappingProxyType[SplitType, Sequence[ScenarioData]]:
        """
        Build per-split ScenarioData for a single NuScenes version (e.g. `v1.0-trainval`).

        Args:
          dataset_params: Dataset parameters, where `dataset_name` is the NuScenes version
            directory name (e.g. `v1.0-trainval`) under `scenario_root_path`.

        Returns:
          MappingProxyType[SplitType, Sequence[ScenarioData]]: Dictionary of SplitType to a list
          of ScenarioData for the corresponding split.
        """

        version_root_path = self.scenario_root_path / dataset_params.dataset_name
        logger.info(f"Loading NuScenes tables from {version_root_path}")
        nusc = NuScenes(
            version=dataset_params.dataset_name,
            dataroot=str(self.scenario_root_path),
            verbose=False,
        )

        scene_name_to_splits = self._build_scene_name_to_splits(dataset_params.dataset_name)

        scenario_splits = defaultdict(list)
        for scene in nusc.scene:
            splits = scene_name_to_splits.get(scene["name"])
            if splits is None:
                continue
            log_record = nusc.get("log", scene["log_token"])
            scenario = ScenarioData(
                dataset_params=dataset_params,
                scenario_id=scene["name"],
                scenario_version=dataset_params.dataset_name,
                vehicle_type=log_record.get("vehicle"),
                location=log_record.get("location"),
            )
            for split in splits:
                scenario_splits[split].append(scenario)
        return scenario_splits

    def _build_scene_name_to_splits(self, dataset_name: str) -> Mapping[str, Sequence[SplitType]]:
        """
        Build a mapping of scene name -> SplitTypes using nuscenes-devkit's official scene-name
        split lists (`nuscenes.utils.splits`) for this NuScenes version.

        Args:
          dataset_name: NuScenes version directory name (e.g. `v1.0-trainval`, `v1.0-mini`),
            used to select which official split lists apply.

        Returns:
          Mapping[str, Sequence[SplitType]]: Dictionary of scene name to the splits it serves.
        """

        split_scene_names = _VERSION_TO_SPLIT_SCENE_NAMES.get(dataset_name)
        if split_scene_names is None:
            raise ValueError(
                f"No known train/val/test split for NuScenes version {dataset_name}; "
                f"expected one of {sorted(_VERSION_TO_SPLIT_SCENE_NAMES.keys())}."
            )

        scene_name_to_splits = defaultdict(list)
        for split, scene_names in split_scene_names.items():
            for scene_name in scene_names:
                scene_name_to_splits[scene_name].append(split)

        return scene_name_to_splits
