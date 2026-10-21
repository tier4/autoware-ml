from __future__ import annotations

import bisect
import copy
import itertools
import time
from abc import abstractmethod
from pathlib import Path
from collections.abc import Sequence
from typing import Self


import polars as pl
from torch.utils.data import Dataset

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider
from autoware_ml.transforms.base import TransformsCompose


class BaseDataset(Dataset):
    """Dataset interface every database specific dataset implements.

    A dataset config names the database it reads, through the root path, and the annotations
    of that corpus supervising the run. The datamodule declares it once, without records, and
    binds it to the records of its dataset source in setup. Binding returns a copy serving
    those records. A dataset built with its records directly is bound from the start.

    The dataset also owns the map provider of its corpus, when the corpus ships lanelet maps.
    The evaluation filters and collision metrics read the map of a scene through it, so the
    provider follows the database the dataset reads instead of the task being evaluated.
    """

    def __init__(
        self,
        database_root_path: str,
        *,
        max_num_3d_gt_bboxes: int | None = None,
        dataset_records_dataframe: pl.DataFrame | None = None,
        transforms: TransformsCompose | None,
        det3d_supervised: bool = True,
        seg3d_supervised: bool = True,
        map_provider: LaneletMapProvider | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
          database_root_path: Root directory of the database the dataset reads.
          max_num_3d_gt_bboxes: Number of 3D boxes every sample is padded to in a batch, extra
              boxes are dropped. Required when the samples carry 3D boxes, None otherwise.
          dataset_records_dataframe: Polars DataFrame of the dataset records. None until the
              dataset is bound to the records of its source.
          transforms: Global transforms to be applied to the dataset records.
          det3d_supervised: Whether the boxes of this corpus are used. If False, every sample
              gets an empty box set.
          seg3d_supervised: Whether the semantic masks of this corpus are used. If False, every
              point gets the ignore index.
          map_provider: Resolves a scene token of this corpus to its lanelet map. None when
              the corpus has no maps, which leaves every map-backed metric filter without a
              provider.
        """
        super().__init__()
        self.database_root_path = Path(database_root_path)
        self.max_num_3d_gt_bboxes = max_num_3d_gt_bboxes
        self.transforms = transforms
        self.dataset_records_dataframe = dataset_records_dataframe
        self.det3d_supervised = det3d_supervised
        self.seg3d_supervised = seg3d_supervised
        self.map_provider = map_provider

    @property
    def is_bound(self) -> bool:
        """Whether the dataset serves records."""
        return self.dataset_records_dataframe is not None

    def bind(self, dataset_records_dataframe: pl.DataFrame) -> Self:
        """
        Return a copy of this dataset serving the given records.

        This dataset stays as declared, so the datamodule binds it to the records of every
        split its source serves.

        Args:
          dataset_records_dataframe: Records of the database of the source in one split.

        Returns:
          Self: Bound copy of the dataset.
        """
        dataset = copy.copy(self)
        dataset._serve_records(dataset_records_dataframe)
        return dataset

    def _serve_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Keep the records to serve. Subclasses override it to prepare their readers.

        Args:
          dataset_records_dataframe: Records of the database of the source in one split.
        """
        self.dataset_records_dataframe = dataset_records_dataframe

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
        if self.dataset_records_dataframe is None:
            raise ValueError(
                "The dataset is not bound to records, so it has no samples. The datamodule "
                "binds it in setup()."
            )
        return len(self.dataset_records_dataframe) * self.samples_per_record

    def scene_index_groups(self) -> list[list[int]]:
        """Sample indices of every scene, each scene in time order.

        Scene streaming feeds a temporal model one scene after another, frame by frame. The
        records are grouped by their scenario and ordered by their sample time, so the row
        order of the record table does not matter.

        Returns:
          list[list[int]]: One list of sample indices per scenario, the scenarios in the order
            they first appear in the record table.

        Raises:
          ValueError: If the dataset is not bound to records, or if every record serves
            several samples, one per lidar or camera source.
        """
        if self.dataset_records_dataframe is None:
            raise ValueError(
                "The dataset is not bound to records, so it has no scenes. The datamodule "
                "binds it in setup()."
            )
        if self.samples_per_record != 1:
            raise ValueError(
                "Scene streaming serves one sample per record, but every record of this dataset "
                f"serves {self.samples_per_record} sources."
            )
        scenario_ids = self.dataset_records_dataframe.get_column(
            DatasetTableSchema.SCENARIO_ID.name
        ).to_list()
        timestamps = self.dataset_records_dataframe.get_column(
            DatasetTableSchema.TIMESTAMP_SECONDS.name
        ).to_list()
        groups: dict[str, list[int]] = {}
        for index, scenario_id in enumerate(scenario_ids):
            groups.setdefault(scenario_id, []).append(index)
        return [sorted(group, key=timestamps.__getitem__) for group in groups.values()]

    def __getitem__(self, index: int) -> ModelGTSample:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed ModelGTSample instance, stamped with its loading time.
        """
        start_time = time.perf_counter()
        model_gt_sample = self.get_data_sample(index)
        transformed_gt_sample = self.apply_transforms(model_gt_sample)
        return transformed_gt_sample._replace(io_processing_time=time.perf_counter() - start_time)

    @abstractmethod
    def get_data_sample(self, index: int) -> ModelGTSample:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            ModelGTSample instance consumed by the transform pipeline.
        """

    def apply_transforms(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Apply the transform pipeline of the dataset to a metadata sample.

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
        # Padding size of the 3D boxes, the largest one a source sets
        self.max_num_3d_gt_bboxes = max(
            (d.max_num_3d_gt_bboxes for d in self.datasets if d.max_num_3d_gt_bboxes is not None),
            default=None,
        )

    def __len__(self) -> int:
        """Return the number of samples of the split, repeated sources included.

        Returns:
          int: Number of samples.
        """
        return self.ends[-1]

    def scene_index_groups(self) -> list[list[int]]:
        """Split indices of every scene of every source, each scene in time order.

        Every repeat of a source serves its scenes once more, as scenes of their own.

        Returns:
          list[list[int]]: One list of split indices per scene and repeat, the sources in
            declaration order.
        """
        groups: list[list[int]] = []
        start = 0
        for dataset, repeat in zip(self.datasets, self.repeats, strict=True):
            source_groups = dataset.scene_index_groups()
            for _ in range(repeat):
                groups.extend([[start + index for index in group] for group in source_groups])
                start += len(dataset)
        return groups

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
