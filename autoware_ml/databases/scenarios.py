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

from pathlib import Path
from typing import Sequence, Mapping, Annotated

from pydantic import BaseModel, ConfigDict, BeforeValidator, Field, model_validator

from autoware_ml.types.dataset import SplitType, SweepDirection


def path_adapter(path: str | Path) -> Path:
    """
    Adapter for pathlib. If the path is a string, convert it
    to a Path object.

    Args:
      path: Path to be adapted, can be a string or a Path object.

    Returns:
      Path: Adapted path.
    """

    if isinstance(path, str):
        return Path(path)
    return path


PathAdapter = Annotated[Path, BeforeValidator(path_adapter)]


class DatasetParams(BaseModel):
    """
    Parameters for a dataset, for example, the sweep window and sampling steps
    when preprocessing it.

    Attributes:
      dataset_name: Name of the dataset.
      max_past_sweeps: Maximum number of past lidar frames and past camera keyframes recorded
        with a sample.
      max_future_sweeps: Maximum number of lidar frames captured after a sample to record
        with it.
      sample_steps: Number of steps to sample.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    dataset_name: str
    max_past_sweeps: int = Field(ge=0)
    max_future_sweeps: int = Field(ge=0)
    sample_steps: int = Field(ge=1)

    def max_sweeps(self, direction: SweepDirection) -> int:
        """
        Number of sweeps recorded on one side of a sample.

        Args:
          direction: Side of the sample the sweeps are collected from.

        Returns:
          int: Maximum number of sweeps on that side.
        """

        if direction is SweepDirection.PAST:
            return self.max_past_sweeps
        return self.max_future_sweeps

    def __str__(self) -> str:
        """String representation of the dataset parameters."""
        return (
            f"DatasetParams(dataset_name={self.dataset_name}, "
            f"max_past_sweeps={self.max_past_sweeps}, "
            f"max_future_sweeps={self.max_future_sweeps}, "
            f"sample_steps={self.sample_steps})"
        )


class ScenarioData(BaseModel):
    """
    Class to store the scenario data for a single scenario.
    Note that one ScenarioData object can have multiple samples/frames
    in the scenario.

    Attributes:
      dataset_params: Parameters of the dataset the scenario belongs to.
      scenario_id: ID of the scenario.
      scenario_version: Version of the scenario.
      vehicle_type: Type of the vehicle.
      location: Location of the scenario.
    """

    # Set model config to frozen and strict
    model_config = ConfigDict(frozen=True, strict=True)

    dataset_params: DatasetParams
    scenario_id: str
    scenario_version: str
    vehicle_type: str | None = None
    location: str | None = None

    def __str__(self) -> str:
        """
        String representation of the scenario data.

        Returns:
          str: String representation of the scenario data.
        """

        return (
            f"ScenarioData(dataset_params={self.dataset_params}, "
            f"scenario_id={self.scenario_id}, "
            f"scenario_version={self.scenario_version}, "
            f"vehicle_type={self.vehicle_type}, "
            f"location={self.location})"
        )


class Scenarios(BaseModel):
    """
    Scenario datasets class. This class is used to store the scenario data for a dataset.

    Attributes:
      scenario_root_path: Root path where the scenario yaml files are stored.
      dataset_params: Parameters for the dataset.
      scenario_data: Dictionary of split type to a list of ScenarioData.
    """

    # Set model config to frozen and strict
    model_config = ConfigDict(frozen=True, strict=True)

    scenario_root_path: PathAdapter  # Root path where the scenario yaml files are stored
    dataset_params: Sequence[DatasetParams]
    scenario_data: Mapping[SplitType, Sequence[ScenarioData]] | None = None

    def __str__(self) -> str:
        """
        String representation of the scenarios.

        Returns:
          str: String representation of the scenarios.
        """

        string = f"Scenarios(scenario_root_path={str(self.scenario_root_path)}"
        string += "dataset_params=("
        for dataset_param in self.dataset_params:
            string += f"{dataset_param}, "
        string += "), "
        string += "scenario_data=("
        for split, scenario_data in self.scenario_data.items():
            string += f"{split}: {scenario_data}, "
        string += "))"
        return string

    def __eq__(self, other: Scenarios) -> bool:
        """
        Compare two scenarios by their version and scenario IDs.

        Returns:
          bool: True if the scenarios are equal, False otherwise.
        """

        return (
            self.scenario_root_path == other.scenario_root_path
            and self.dataset_params == other.dataset_params
            and self.scenario_data == other.scenario_data
        )

    def __hash__(self) -> int:
        """
        Hash the scenarios by their version and scenario IDs.

        Returns:
          int: Hash of the scenarios.
        """

        return hash(str(self))

    @model_validator(mode="after")
    def build_scenarios(self) -> Scenarios:
        """
        Definition of the logic to build Scenarios for a dataset.

        Returns:
          Scenarios: Scenarios class instance.
        """

        raise NotImplementedError("Subclasses must implement build_scenarios()!")

    def get_all_scenario_data(self) -> Sequence[ScenarioData]:
        """
        Get all scenario data from all splits.

        Returns:
          Sequence[ScenarioData]: Sequence of scenario data.
        """

        return [scenario_data for split in self.scenario_data.values() for scenario_data in split]
