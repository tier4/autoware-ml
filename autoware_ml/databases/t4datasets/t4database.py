from typing import ImmutableMapping, Iterable

from hydra.utils import instantiate
from omegaconf import DictConfig

from autoware_ml.common.enums import TaskType
from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.t4datasets.t4scenarios import T4Scenarios


class T4Database(BaseDatabase):
    """
    T4 database class.
    """

    def __init__(
        self,
        database_version: str,
        database_root_path: str,
        scenario_root_path: str,
        scenario_configs: ImmutableMapping[str, DictConfig],
        task_types: Iterable[TaskType],
        main_database: str,
    ) -> None:
        """Initialize T4 database. Please refer to the BaseDatabase class for more details."""
        super().__init__(
            database_version=database_version,
            database_root_path=database_root_path,
            scenario_root_path=scenario_root_path,
            scenario_configs=scenario_configs,
            task_types=task_types,
            main_database=main_database,
        )
        self.scenarios: ImmutableMapping[str, T4Scenarios] = {
            scenario_group_name: instantiate(scenario_config, root_path=self.database_root_path)
            for scenario_group_name, scenario_config in scenario_configs.items()
        }
