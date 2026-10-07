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
        max_num_3d_gt_bboxes: int,
        dataset_records_dataframe: pl.DataFrame,
        transforms: TransformsCompose | None,
        det3d_supervised: bool,
        seg3d_supervised: bool,
    ) -> None:
        """
        Initialize the dataset.

        Args:
          database_root_path: Root directory of the dataset.
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
              This is allowed to be 0 if the dataset does not contain any 3D ground truth
              bounding boxes or it does not need to run 3D detection tasks.
          dataset_records_dataframe: Polars DataFrame of the dataset records.
          transforms: Global transforms to be applied to the dataset records.
          det3d_supervised: Whether the boxes of this corpus are used. If False, every sample
              gets an empty box set.
          seg3d_supervised: Whether the semantic masks of this corpus are used. If False, every
              point gets the ignore index.
        """
        super().__init__()
        self.database_root_path = Path(database_root_path)
        self.max_num_3d_gt_bboxes = max_num_3d_gt_bboxes
        self.transforms = transforms
        self.dataset_records_dataframe = dataset_records_dataframe
        self.det3d_supervised = det3d_supervised
        self.seg3d_supervised = seg3d_supervised

    @property
    def samples_per_record(self) -> int:
        """Number of samples every record serves."""
        return 1

    def record_index(self, sample_index: int) -> int:
        """Index of the record a sample is served from.

        Args:
          sample_index: Sample index.

        Returns:
          int: Record index.
        """
        return sample_index // self.samples_per_record

    def __len__(self) -> int:
        """Return the number of samples, one per record and served source.

        Returns:
          int: Number of samples.
        """
        return len(self.dataset_records_dataframe) * self.samples_per_record

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


class ConcatDataset(Dataset):
    """Serve one split assembled from several corpora.

    Every index maps to one sample of one source. A source with repeat N contributes its
    samples N times, so a small corpus can keep a share of the epoch next to a large one.
    """

    def __init__(self, datasets: Sequence[BaseDataset], repeats: Sequence[int]) -> None:
        """
        Initialize the concatenated dataset.

        Args:
          datasets: Dataset of every source of the split, in declaration order.
          repeats: How many times each source contributes its samples to one epoch.
        """
        super().__init__()
        if len(datasets) != len(repeats):
            raise ValueError(
                f"Expected one repeat per source, got {len(datasets)} sources and "
                f"{len(repeats)} repeats."
            )
        if not len(datasets):
            raise ValueError("A concatenated dataset needs at least one source.")
        self.datasets = tuple(datasets)
        self.repeats = tuple(repeats)
        # End index of the block of every source, its samples times its repeat
        self.ends = tuple(
            itertools.accumulate(
                len(dataset) * repeat for dataset, repeat in zip(datasets, repeats, strict=True)
            )
        )
        self.max_num_3d_gt_bboxes = max(dataset.max_num_3d_gt_bboxes for dataset in self.datasets)

    def __len__(self) -> int:
        """Return the number of samples of the split, repeated sources included.

        Returns:
          int: Number of samples.
        """
        return self.ends[-1]

    def locate(self, index: int) -> tuple[int, int]:
        """Find the source and the sample behind one index of the split.

        Args:
          index: Sample index of the split.

        Returns:
          tuple[int, int]: Index of the source and index of the sample in that source.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Sample index {index} is out of range for {len(self)} samples.")
        source_index = bisect.bisect_right(self.ends, index)
        start = self.ends[source_index - 1] if source_index else 0
        return source_index, (index - start) % len(self.datasets[source_index])

    def __getitem__(self, index: int) -> ModelGTSample:
        """Load and transform the sample behind one index of the split.

        Args:
            index: Sample index.

        Returns:
            Transformed ModelGTSample instance.
        """
        source_index, sample_index = self.locate(index)
        return self.datasets[source_index][sample_index]

    def collate_fn(self, batch: Sequence[ModelGTSample]) -> ModelGTBatch:
        """
        Collate a batch of ModelGTSample into a ModelGTBatch.

        A batch mixes the sources of the split, so the boxes are padded to the largest budget
        of any source.

        Args:
          batch: List of ModelGTSample instances to be collated.

        Returns:
          ModelGTBatch: Collated GT batch.
        """
        return ModelGTBatch.collate_gt_samples(
            gt_samples=batch, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )
