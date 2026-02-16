"""
Dataset generator basic interface.
"""

import logging

from pathlib import Path
from typing import Iterable

import polars as pl

from autoware_ml.common.enums import TaskType
from autoware_ml.annotations.schemas.schemas import AnnotationTableSchema
from autoware_ml.annotations.schemas.records import AnnotationTableRecord

logger = logging.getLogger(__name__)


class DatasetAnnotationGenerator:
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
        max_sweeps: int = 0,
    ) -> None:
        """
        :param root_path: Root path of the dataset.
        :param out_dir: Output directory for annotations.
        :param database_version: Version of the database.
        :param task_types: List of task types to generate annotations for.
        :param output_file_postfix: Postfix for the output file name.
        :param max_sweeps: Max number of lidar sweeps to include, only for 3D.
        :return: None
        """
        self.root_path = Path(root_path)
        self.out_dir = Path(out_dir)
        self.database_version = database_version
        self.max_sweeps = max_sweeps
        self.task_types = task_types
        self.output_file_postfix = output_file_postfix

        # Create out directory if it doesn't exist
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        return f"Root path: {self.root_path}, out dir: {self.out_dir}, database version: {self.database_version}, task types: {self.task_types}, output file postfix: {self.output_file_postfix}, max sweeps: {self.max_sweeps}"

    @property
    def dataset_type(self) -> str:
        raise NotImplementedError("Subclasses must define dataset_type!")

    @property
    def annotation_file_name(self) -> str:
        """Output annotation file name."""
        return f"{self.dataset_type}_annotations_{self.database_version}_{self.output_file_postfix}.parquet"

    def generate_annotation_records(self) -> Iterable[AnnotationTableRecord]:
        """
        Generate annotation table data records.
        """
        raise NotImplementedError("Subclasses must implement generate_annotation_records method!")

    def run(self) -> None:
        """
        Run annotation generator.
        """
        logger.info(f"Running {self.dataset_type} dataset annotation generator with {self}")
        # Generate annotations
        annotation_records = self.generate_annotation_records()

        # Save annotation records
        self.save_annotations(annotation_records)

        logger.info(
            f"{self.dataset_type} dataset annotation generator completed successfully with {len(annotation_records)} annotation frames/records, and saved to {self.out_dir / self.annotation_file_name}"
        )

    def save_annotation_records(self, annotation_records: Iterable[AnnotationTableRecord]) -> None:
        """
        Save annotations to a polars .parquet file.
        :param annotation_records: List(row) of annotation table data records.
        :return: None
        """
        # Convert annotations to a polars DataFrame
        df = pl.DataFrame(annotation_records, schema=AnnotationTableSchema.to_polars_schema())

        # Save to a .parquet file
        df.write_parquet(self.out_dir / self.annotation_file_name)
