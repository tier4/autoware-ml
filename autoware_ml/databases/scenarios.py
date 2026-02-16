from pathlib import Path
from typing import Iterable, ImmutableMapping

from pydantic import BaseModel, ConfigDict, model_validator

from autoware_ml.common.enums import SplitType


class ScenarioData(BaseModel):
		"""
		Scenario identifier. This is the unique identifier for a scenario.
		"""
		# Set model config to frozen and strict
		model_config = ConfigDict(frozen=True, strict=True)
		
		db_version: str
		scenario_id: str
		version: str
		vehicle_type: str | None = None
		location: str | None = None


class Scenarios(BaseModel):
		"""
		Scenario datasets class.
		"""
		# Set model config to frozen and strict
		model_config = ConfigDict(frozen=True, strict=True)

		version: str
		scenario_root_path: Path    # Root path where the scenario yaml files are stored
		db_versions: Iterable[str]
		scenario_data: ImmutableMapping[SplitType, Iterable[ScenarioData]] | None = None # {SplitType: List[ScenarioData]}

		@model_validator(mode='after')
		def build_scenarios(self) -> None:
				"""
				Build scenario data.
				"""
				raise NotImplementedError("Subclasses must implement build_scenario_data()!")
