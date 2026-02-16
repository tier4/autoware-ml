from pathlib import Path
from typing import ImmutableMapping, Iterable

from omegaconf import DictConfig

from autoware_ml.common.enums import TaskType
from autoware_ml.databases.scenarios import Scenarios

class BaseDatabase:
    """
    Interface for database classes.
    """
    def __init__(
        self, 
        database_version: str, 
        database_root_path: str,
        scenario_root_path: str,
        scenario_configs: ImmutableMapping[str, DictConfig], 
        task_types: Iterable[TaskType],
        main_database: str) -> None:
        """
        Initialize database interface.
        :param database_version: Version of the database.
        :param database_root_path: Root path where the actual annotation files are stored.
        :param scenario_root_path: Root path where the scenario yaml files are stored.
        :param scenario_configs: Scenario configurations for each 
          scenario in {'scenario_group_name': scenario_config}.
        :param task_types: Supported task types.
        :param main_database: Main database/scenario group name.
        """
        self.database_version = database_version
        self.database_root_path = Path(database_root_path)
        self.scenario_root_path = Path(scenario_root_path)
        self.main_database = main_database
        self.scenario_configs = scenario_configs
        self.task_types = task_types
        self.scenarios = {}
    
    @property
    def datatabase_version(self) -> str:
        """ Get the version of the database. """
        return self.database_version
      
    @property
    def database_root_path(self) -> Path:
        """ Get the root path of the database. """
        return self.database_root_path

    @property
    def scenario_root_path(self) -> Path:
        """ Get the root path of the scenario files. """
        return self.scenario_root_path

    @property
    def main_database(self) -> str:
        """ Get the main database/scenario group name. """
        return self.main_database

    @property
    def scenarios(self) -> ImmutableMapping[str, Scenarios]:
        """ Get the scenarios for each scenario group. """
        return self.scenarios
    
    @property
    def task_types(self) -> Iterable[TaskType]:
        """ Get the task types supported by the database. """
        return self.task_types
