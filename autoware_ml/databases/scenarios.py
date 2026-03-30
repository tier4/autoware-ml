from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Annotated

from pydantic import BaseModel, ConfigDict, BeforeValidator, model_validator

from autoware_ml.common.enums.enums import SplitType


def path_adapter(path: str | Path) -> Path:
    """Adapter for pathlib."""
    if isinstance(path, str):
        return Path(path)
    return path


PathAdapter = Annotated[Path, BeforeValidator(path_adapter)]


class DatabaseVersion(BaseModel):
    """Database version and its settings."""

    db_version: str
    max_sweeps: int
    sample_steps: int

    def __eq__(self, other: DatabaseVersion) -> bool:
        """Compare two database versions by their version and settings."""
        return (
            self.db_version == other.db_version
            and self.max_sweeps == other.max_sweeps
            and self.sample_steps == other.sample_steps
        )

    def __hash__(self) -> int:
        """Hash the database version by its version and settings."""
        return hash((self.db_version, self.max_sweeps, self.sample_steps))


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
        return hash(
            (
                self.db_version,
                self.scenario_id,
                self.scenario_version,
                self.max_sweeps,
                self.sample_steps,
                self.vehicle_type,
                self.location,
            )
        )


class Scenarios(BaseModel):
    """
    Scenario datasets class.
    """

    # Set model config to frozen and strict
    model_config = ConfigDict(frozen=True, strict=True)

    version: str
    scenario_root_path: PathAdapter  # Root path where the scenario yaml files are stored
    db_versions: Iterable[DatabaseVersion]
    scenario_data: Mapping[SplitType, Iterable[ScenarioData]] | None = None

    def __eq__(self, other: Scenarios) -> bool:
        """Compare two scenarios by their version and scenario IDs."""
        return (
            self.version == other.version
            and self.scenario_root_path == other.scenario_root_path
            and self.db_versions == other.db_versions
            and self.scenario_data == other.scenario_data
        )

    def __hash__(self) -> int:
        """Hash the scenarios by their version and scenario IDs."""
        hash_attributes = (
            self.version,
            self.scenario_root_path,
        )

        hash_attributes += tuple(
            self.db_versions,
        )
        # For dictionary, we need to hash the dictionary keys and values
        for split, scenario_data in self.scenario_data.items():
            hash_attributes += (
                split,
                tuple(scenario_data),
            )
        return hash(hash_attributes)

    @model_validator(mode="after")
    def build_scenarios(self) -> None:
        """
        Build scenario data.
        """
        raise NotImplementedError("Subclasses must implement build_scenario_data()!")

    def get_all_scenario_data(self) -> Iterable[ScenarioData]:
        """Get all scenario data."""
        return [scenario_data for split in self.scenario_data.values() for scenario_data in split]
