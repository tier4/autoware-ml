import logging

from typing import Iterable

from autoware_ml.common.enums.enums import TaskType
from autoware_ml.datasets.generators.dataset_generator import DatasetGenerator
from autoware_ml.datasets.schemas.records import DatasetTableRecord

logger = logging.getLogger(__name__)


class T4DatasetGenerator(DatasetGenerator):
    """T4 dataset generator."""

    def __init__(
        self,
        root_path: str,
        out_dir: str,
        database_version: str,
        task_types: Iterable[TaskType],
        output_file_postfix: str,
        max_sweeps: int = 0,
    ):
        super().__init__(
            root_path=root_path,
            out_dir=out_dir,
            database_version=database_version,
            task_types=task_types,
            output_file_postfix=output_file_postfix,
            max_sweeps=max_sweeps,
        )

    @property
    def dataset_type(self) -> str:
        return "t4dataset"

    def generate_dataset_records(self) -> Iterable[DatasetTableRecord]:
        """Generate dataset table data records."""
        # TODO: Implement T4 dataset annotation table data records generation
        raise NotImplementedError
