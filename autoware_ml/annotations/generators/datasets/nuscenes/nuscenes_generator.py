import logging

from typing import Iterable

from autoware_ml.annotations.enums.enums import TaskType
from autoware_ml.annotations.generators.generator import DatasetAnnotationGenerator
from autoware_ml.annotations.schemas.records import AnnotationTableRecord

logger = logging.getLogger(__name__)


class NuScenesAnnotationGenerator(DatasetAnnotationGenerator):
    """
    Dataset annotation generator basic interface.
    """

    def __init__(
        self,
        root_path: str,
        out_dir: str,
        database_version: str,
        task_types: Iterable[TaskType],
        output_file_postfix: str,
    ):
        max_sweeps: int = (0,)
        """"""
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
        return "nuscenes"

    def generate_annotation_records(self) -> Iterable[AnnotationTableRecord]:
        """
        Generate annotation table data records.
        """
        # TODO: Implement NuScenes dataset annotation table data records generation
        raise NotImplementedError
