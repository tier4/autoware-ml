from collections import defaultdict
import yaml 
from typing import Iterable, ImmutableMapping

from pydantic import model_validator

from autoware_ml.common.enums import SplitType
from autoware_ml.databases.scenarios import ScenarioData, Scenarios


class T4Scenarios(Scenarios):
		""" T4 scenario datasets class. """
		
		@model_validator(mode='after')
		def build_scenarios(self) -> None:
				""" Build scenarios from database scenarios, and overwrite the scenario_data attribute. """
				scenario_data = defaultdict(list)
				for db_version in self.db_versions:
						db_yaml_path = self.scenario_root_path / db_version / '.yaml'
						with open(db_yaml_path, "r") as f:
								db_scenarios: ImmutableMapping[str, Iterable[str]] = yaml.safe_load(f)
						
						scenario_splits = self._build_scenario_splits(db_scenarios, db_version)
						for split, scenarios in scenario_splits.items():
								scenario_data[split] += scenarios
				
				object.__setattr__(self, 'scenario_data', scenario_data)
	
		@staticmethod
		def _build_scenario_data(scenario_id: str, db_version: str) -> ScenarioData:
				""" 
				Build scenario data from a scenario ID and a database version.
				:param scenario_id: Scenario ID.
				:param db_version: Database version.
				:return: Scenario data.
				"""
				dataset_scene_info = scenario_id.split("/")
				if len(dataset_scene_info) == 4:
						scenario_id, version, city, vehicle_type = dataset_scene_info
				elif len(dataset_scene_info) == 2:
						scenario_id, version = dataset_scene_info
						city = vehicle_type = None
				else:
						raise ValueError(f"Invalid scenario ID: {scenario_id}")
				
				return ScenarioData(
					db_version=db_version,
					scenario_id=scenario_id,
					version=version,
					vehicle_type=vehicle_type,
					location=city
				)

		def _build_scenario_splits(
				self, 
				db_scenarios: ImmutableMapping[str, Iterable[str]], 
				db_version: str
				) -> ImmutableMapping[SplitType, Iterable[ScenarioData]]:
				""" 
				Build splits from a database scenarios.
				:param db_scenarios: Database scenarios.
				:param db_version: Database version.
				:return: List of ScenarioData for each split in {SplitType: List[ScenarioData]}.
				"""
				scenario_splits = {}
				for split in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST]:
						selected_scenarios = db_scenarios.get(split, [])
						scenario_splits[split] = [self._build_scenario_data(scenario_id, db_version) for scenario_id in selected_scenarios]
				return scenario_splits
