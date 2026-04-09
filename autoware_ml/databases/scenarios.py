from __future__ import annotations

from pathlib import Path
from typing import Sequence, Mapping, Annotated

from pydantic import BaseModel, ConfigDict, BeforeValidator, model_validator

from autoware_ml.common.enums.enums import SplitType


def path_adapter(path: str | Path) -> Path:
    """Adapter for pathlib."""
    if isinstance(path, str):
        return Path(path)
    return path


PathAdapter = Annotated[Path, BeforeValidator(path_adapter)]


class DatasetParams(BaseModel):
    """Parameters for a dataset, for example, version, max_sweeps, and sampling steps."""

    model_config = ConfigDict(frozen=True, strict=True)

    dataset_name: str
    max_sweeps: int
    sample_steps: int

    def __str__(self) -> str:
        """String representation of the database version."""
        return (
            f"DatasetParams(dataset_name={self.dataset_name}, "
            f"max_sweeps={self.max_sweeps}, "
            f"sample_steps={self.sample_steps})"
        )

    def __eq__(self, other: DatasetParams) -> bool:
        """Compare two database versions by their version and settings."""
        return (
            self.dataset_name == other.dataset_name
            and self.max_sweeps == other.max_sweeps
            and self.sample_steps == other.sample_steps
        )

    def __hash__(self) -> int:
        """Hash the database version by its version and settings."""
        return hash(str(self))


class ScenarioData(BaseModel):
    """
    Scenario identifier. This is the unique identifier for a scenario.
    """

    # Set model config to frozen and strict
    model_config = ConfigDict(frozen=True, strict=True)

    db_version: str
    scenario_id: str
    scenario_version: str
    max_sweeps: int
    sample_steps: int
    vehicle_type: str | None = None
    location: str | None = None

    def __str__(self) -> str:
        """String representation of the scenario data."""
        return (
            f"ScenarioData(db_version={self.db_version}, "
            f"scenario_id={self.scenario_id}, "
            f"scenario_version={self.scenario_version}, "
            f"max_sweeps={self.max_sweeps}, "
            f"sample_steps={self.sample_steps}, "
            f"vehicle_type={self.vehicle_type}, "
            f"location={self.location})"
        )

    def __eq__(self, other: ScenarioData) -> bool:
        """Compare two scenario data by their version and scenario IDs."""
        return (
            self.db_version == other.db_version
            and self.scenario_id == other.scenario_id
            and self.scenario_version == other.scenario_version
            and self.max_sweeps == other.max_sweeps
            and self.sample_steps == other.sample_steps
            and self.vehicle_type == other.vehicle_type
            and self.location == other.location
        )

    def __hash__(self) -> int:
        """Hash the scenario data by its version and scenario IDs."""
        return hash(str(self))


class Scenarios(BaseModel):
    """
    Scenario datasets class.
    """

    # Set model config to frozen and strict
    model_config = ConfigDict(frozen=True, strict=True)

    version: str
    scenario_root_path: PathAdapter  # Root path where the scenario yaml files are stored
    dataset_params: Sequence[DatasetParams]
    scenario_data: Mapping[SplitType, Sequence[ScenarioData]] | None = None

    def __str__(self) -> str:
        """String representation of the scenarios."""
        string = (
            f"Scenarios(version={self.version}, scenario_root_path={str(self.scenario_root_path)}"
        )
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
        """Compare two scenarios by their version and scenario IDs."""
        return (
            self.version == other.version
            and self.scenario_root_path == other.scenario_root_path
            and self.dataset_params == other.dataset_params
            and self.scenario_data == other.scenario_data
        )

    def __hash__(self) -> int:
        """Hash the scenarios by their version and scenario IDs."""
        return hash(str(self))

    @model_validator(mode="after")
    def build_scenarios(self) -> Scenarios:
        """
        Build scenario data.
        """
        raise NotImplementedError("Subclasses must implement build_scenario_data()!")

    def get_all_scenario_data(self) -> Sequence[ScenarioData]:
        """Get all scenario data."""
        return [scenario_data for split in self.scenario_data.values() for scenario_data in split]
