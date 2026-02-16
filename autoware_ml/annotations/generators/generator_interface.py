from abc import abstractmethod
from typing import Iterable, Protocol

from autoware_ml.annotations.schemas.records import AnnotationTableRecord


class DatasetAnnotationGeneratorInterface(Protocol):
    """Dataset annotation generator interface."""

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Dataset type."""
        raise NotImplementedError("AnnotationGenerator must define dataset_type!")

    @property
    @abstractmethod
    def annotation_file_name(self) -> str:
        """Output annotation file name."""
        raise NotImplementedError("AnnotationGenerator must define annotation_file_name!")

    @abstractmethod
    def generate_annotation_records(self) -> Iterable[AnnotationTableRecord]:
        """
        Generate annotation table data records.
        """
        raise NotImplementedError(
            "AnnotationGenerator must implement generate_annotation_records method!"
        )

    @abstractmethod
    def run(self) -> None:
        """Run annotation generator."""
        raise NotImplementedError("AnnotationGenerator must implement run method!")

    @abstractmethod
    def save_annotation_records(self, annotation_records: Iterable[AnnotationTableRecord]) -> None:
        """
        Save annotations to a polars .parquet file.
        :param annotation_records: List(row) of annotation table data records.
        :return: None
        """
        raise NotImplementedError(
            "AnnotationGenerator must implement save_annotation_records method!"
        )
