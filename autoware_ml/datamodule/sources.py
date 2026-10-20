# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset sources a datamodule is assembled from."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.datamodule.base_dataset import BaseDataset
from autoware_ml.types.dataset import SplitType


@dataclass(frozen=True)
class DatasetSource:
    """One database and the dataset of every split it serves.

    A source serves the splits it declares a dataset for. Every dataset reads the database of
    the source, its config names that database's root path, and declares which annotations of
    the corpus supervise the run. The datamodule binds a copy of the dataset to the records of
    the database in each split.

    Attributes:
      database: Database providing the dataset records of the source.
      train_dataset: Dataset of the training split, None when the source stays out of it.
      validation_dataset: Dataset of the validation split, None when the source stays out of it.
      test_dataset: Dataset of the test split, None when the source stays out of it.
      predict_dataset: Dataset of the predict split, None when the source stays out of it.
      repeat: How many times the frames of the source appear in one training epoch.
    """

    database: BaseDatabase
    train_dataset: BaseDataset | None = None
    validation_dataset: BaseDataset | None = None
    test_dataset: BaseDataset | None = None
    predict_dataset: BaseDataset | None = None
    repeat: int = 1

    def __post_init__(self) -> None:
        """Validate the source declaration."""
        if not len(self.datasets):
            raise ValueError("A dataset source needs a dataset for at least one split.")
        root_path = Path(self.database.root_path)
        for split, dataset in self.datasets.items():
            if dataset.database_root_path != root_path:
                raise ValueError(
                    f"The {split.value} dataset of the source reads {dataset.database_root_path} "
                    f"but its database {self.database.version} is at {root_path}. Compose a "
                    "dataset config for that database."
                )
        if self.repeat < 1:
            raise ValueError(f"A dataset source repeat must be at least 1, got {self.repeat}.")
        if self.repeat != 1 and self.train_dataset is None:
            raise ValueError(
                "A dataset source repeat only balances the training split, but the source "
                "declares no training dataset."
            )

    def _declared_datasets(self) -> tuple[tuple[SplitType, BaseDataset | None], ...]:
        """Dataset field of every split, declared or not."""
        return (
            (SplitType.TRAIN, self.train_dataset),
            (SplitType.VAL, self.validation_dataset),
            (SplitType.TEST, self.test_dataset),
            (SplitType.PREDICT, self.predict_dataset),
        )

    @property
    def datasets(self) -> Mapping[SplitType, BaseDataset]:
        """Dataset of every split the source serves."""
        return MappingProxyType(
            {split: dataset for split, dataset in self._declared_datasets() if dataset is not None}
        )

    def dataset(self, split: SplitType) -> BaseDataset | None:
        """
        Dataset the source serves one split with.

        Args:
          split: Split asked for.

        Returns:
          BaseDataset | None: The declared dataset, None when the source stays out of the split.
        """
        return self.datasets.get(split)
