from abc import abstractmethod
from typing import Iterable, Protocol

from autoware_ml.datasets.schemas.records import DatasetTableRecord


class DatasetGeneratorInterface(Protocol):
    """Dataset generator interface."""

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Dataset type."""
        raise NotImplementedError("DatasetGenerator must define dataset_type!")

    @property
    @abstractmethod
    def dataset_file_name(self) -> str:
        """Output dataset file name."""
        raise NotImplementedError("DatasetGenerator must define dataset_file_name!")

    @abstractmethod
    def generate_dataset_records(self) -> Iterable[DatasetTableRecord]:
        """Generate dataset table data records."""
        raise NotImplementedError(
            "DatasetGenerator must implement generate_dataset_records method!"
        )

    @abstractmethod
    def run(self) -> None:
        """Run dataset generator."""
        raise NotImplementedError("DatasetGenerator must implement run method!")

    @abstractmethod
    def save_dataset_records(self, dataset_records: Iterable[DatasetTableRecord]) -> None:
        """
        Save dataset records to a polars .parquet file.
        :param dataset_records: List(row) of dataset table data records.
        :return: None
        """
        raise NotImplementedError("DatasetGenerator must implement save_dataset_records method!")
