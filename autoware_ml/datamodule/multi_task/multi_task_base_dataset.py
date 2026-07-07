from abc import abstractmethod
from typing import Sequence


import polars as pl
from torch.utils.data import Dataset

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
    MultiTaskGTBatch,
)
from autoware_ml.transforms.multi_task.base import MultiTaskTransformsCompose


class MultiTaskBaseDataset(Dataset):
    """Multi-task dataset interface that can be shared by multiple databases."""

    def __init__(
        self,
        max_num_3d_gt_bboxes: int,
        dataset_records_dataframe: pl.DataFrame | None,
        transforms: MultiTaskTransformsCompose | None,
    ) -> None:
        """
        Initialize the multi-task dataset interface.
        Args:
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
              This is allowed to be 0 if the dataset does not contain any 3D ground truth
              bounding boxes or it does not need to run 3D detection tasks.
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in the
              multi-task dataset. Accept None if the dataset records
              are not available at initialization.
          transforms: Global transforms to be applied to the dataset records.
        """
        super().__init__()
        self.max_num_3d_gt_bboxes = max_num_3d_gt_bboxes
        self.transforms = transforms
        self.dataset_records_dataframe = dataset_records_dataframe

    def __len__(self) -> int:
        """Return the number of dataset records.

        Returns:
          int: Number of dataset records.
        """
        if self.dataset_records_dataframe is None:
            raise ValueError("Dataset records dataframe is not available.")
        return len(self.dataset_records_dataframe)

    def __getitem__(self, index: int) -> MultiTaskGTSample:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed MultiTaskGTSample instance.
        """
        multi_task_gt_sample = self.get_data_sample(index)
        return self.apply_transforms(multi_task_gt_sample)

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """Assign the dataset records dataframe.

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        self.dataset_records_dataframe = dataset_records_dataframe

    @abstractmethod
    def get_data_sample(self, index: int) -> MultiTaskGTSample:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            MultiTaskGTSample instance consumed by the transform pipeline.
        """
        raise NotImplementedError("Dataset must implement get_data_sample")

    def apply_transforms(
        self,
        multi_task_gt_sample: MultiTaskGTSample,
    ) -> MultiTaskGTSample:
        """Apply a specific transform pipeline to a metadata sample.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance.

        Returns:
            Transformed MultiTaskGTSample instance.
        """
        if self.transforms is None:
            return multi_task_gt_sample
        return self.transforms(multi_task_gt_sample)

    def collate_fn(self, batch: Sequence[MultiTaskGTSample]) -> MultiTaskGTBatch:
        """
        Collate a batch of MultiTaskGTSample into a MultiTaskGTBatch.
        Args:
          batch: List of MultiTaskGTSample instances to be collated.
        Returns:
          MultiTaskGTBatch: Collated multi-task GT batch.
        """
        return MultiTaskGTBatch.collate_gt_samples(
            gt_samples=batch, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )
