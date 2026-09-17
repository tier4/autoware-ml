import bisect
import itertools
from abc import abstractmethod
from pathlib import Path
from typing import Sequence


import polars as pl
from torch.utils.data import Dataset

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.transforms.base import TransformsCompose


class BaseDataset(Dataset):
    """Dataset interface every database specific dataset implements."""

    def __init__(
        self,
        database_root_path: str,
        *,
        max_num_3d_gt_bboxes: int | None = None,
        dataset_records_dataframe: pl.DataFrame,
        transforms: TransformsCompose | None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
          database_root_path: Root directory of the dataset.
          max_num_3d_gt_bboxes: Number of 3D boxes every sample is padded to in a batch, extra
              boxes are dropped. Required when the samples carry 3D boxes, None otherwise.
          dataset_records_dataframe: Polars DataFrame of the dataset records.
          transforms: Global transforms to be applied to the dataset records.
        """
        super().__init__()
        self.database_root_path = Path(database_root_path)
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

    def __getitem__(self, index: int) -> ModelGTSample:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed ModelGTSample instance.
        """
        model_gt_sample = self.get_data_sample(index)
        return self.apply_transforms(model_gt_sample)

    @abstractmethod
    def get_data_sample(self, index: int) -> ModelGTSample:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            ModelGTSample instance consumed by the transform pipeline.
        """

    def apply_transforms(
        self,
        model_gt_sample: ModelGTSample,
    ) -> ModelGTSample:
        """Apply a specific transform pipeline to a metadata sample.

        Args:
            model_gt_sample: ModelGTSample instance.

        Returns:
            Transformed ModelGTSample instance.
        """
        if self.transforms is None:
            return model_gt_sample
        return self.transforms(model_gt_sample)

    def collate_fn(self, batch: Sequence[ModelGTSample]) -> ModelGTBatch:
        """
        Collate a batch of ModelGTSample into a ModelGTBatch.

        Args:
          batch: List of ModelGTSample instances to be collated.

        Returns:
          ModelGTBatch: Collated GT batch.
        """
        return ModelGTBatch.collate_gt_samples(
            gt_samples=batch, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )
