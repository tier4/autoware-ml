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
import pickle
from collections import defaultdict
from typing import Mapping, Sequence
from types import MappingProxyType

from nuscenes.nuscenes import NuScenes
from pydantic import model_validator

from autoware_ml.types.dataset import SplitType
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData, Scenarios

logger = logging.getLogger(__name__)

# Maps SplitType to the `nuscenes_infos_<name>.pkl` filename suffix that carries that split's
# scene membership (mmdet3d-style info files, e.g. `nuscenes_infos_train.pkl`).
_SPLIT_INFO_FILE_SUFFIX = {
    SplitType.TRAIN: "train",
    SplitType.VAL: "val",
    SplitType.TEST: "test",
}


class NuScenesScenarios(Scenarios):
    """
    NuScenesScenarios class inherits from Scenarios and defines the logic for building scenario
    data for a NuScenesDataset, where one scenario corresponds to one NuScenes scene.

    Unlike T4Scenarios (which reads per-dataset scenario YAML files listing scene IDs per split),
    NuScenes ships one flat table set per version (e.g. `v1.0-trainval` holds all 850 scenes
    together). Train/val/test split membership is instead derived from the scene_token set found
    in the existing `nuscenes_infos_<split>.pkl` info files (mmdet3d-style), which are already the
    authoritative split for this dataset copy.
    """

    @model_validator(mode="after")
    def build_scenarios(self) -> NuScenesScenarios:
        """
        Build scenarios by loading the NuScenes version tables via nuscenes-devkit, and splitting
        scenes into train/val/test according to the scene_token membership found in each
        `nuscenes_infos_<split>.pkl` file.

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

        scene_token_to_split = self._build_scene_token_to_split(dataset_params.dataset_name)

        scenario_splits = defaultdict(list)
        for scene in nusc.scene:
            split = scene_token_to_split.get(scene["token"])
            if split is None:
                # Scene not present in any known split's info pkl (e.g. v1.0-mini scenes,
                # or a version without info pkls for every split); skip it.
                continue
            log_record = nusc.get("log", scene["log_token"])
            scenario_splits[split].append(
                ScenarioData(
                    dataset_name=dataset_params.dataset_name,
                    scenario_id=scene["name"],
                    scenario_version=dataset_params.dataset_name,
                    vehicle_type=log_record.get("vehicle"),
                    location=log_record.get("location"),
                    max_sweeps=dataset_params.max_sweeps,
                    sample_steps=dataset_params.sample_steps,
                )
            )
        return scenario_splits

    def _build_scene_token_to_split(self, dataset_name: str) -> Mapping[str, SplitType]:
        """
        Build a mapping of scene_token -> SplitType by reading scene membership out of each
        `nuscenes_infos_<split>.pkl` file found under `scenario_root_path`.

        Args:
          dataset_name: NuScenes version directory name, only used for logging.

        Returns:
          Mapping[str, SplitType]: Dictionary of scene_token to the split it belongs to.
        """

        scene_token_to_split = {}
        for split, suffix in _SPLIT_INFO_FILE_SUFFIX.items():
            info_path = self.scenario_root_path / f"nuscenes_infos_{suffix}.pkl"
            if not info_path.exists():
                logger.warning(
                    f"No info pkl found at {info_path} for split {split} "
                    f"of dataset {dataset_name}; scenes for this split will be skipped."
                )
                continue

            with open(info_path, "rb") as f:
                info = pickle.load(f)

            # The test split's info pkl is trimmed (no GT), so `scene_token` may be absent from
            # its records; skip records that lack it rather than failing the whole split.
            scene_tokens = {
                record["scene_token"] for record in info["data_list"] if "scene_token" in record
            }
            if len(scene_tokens) == 0 and len(info["data_list"]) > 0:
                logger.warning(
                    f"Info pkl {info_path} for split {split} has no scene_token field in its "
                    f"records; scenes for this split cannot be determined and will be skipped."
                )
            for scene_token in scene_tokens:
                scene_token_to_split[scene_token] = split

        return scene_token_to_split
