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

"""Training split of a datamodule, the data the anchor recipes read."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial

import polars as pl

from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.sources import DatasetSource, split_database_records
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType
from autoware_ml.types.tasks import TaskType


def describe(value: object) -> str:
    """Canonical text of a dataset task setting, part of the anchor cache key.

    Args:
        value: Task builder, setting or nested value.

    Returns:
        The same text for equal settings in every process.

    Raises:
        TypeError: If the value has no canonical text.
    """
    if isinstance(value, partial):
        func = f"{value.func.__module__}.{value.func.__qualname__}"
        arguments = [describe(item) for item in value.args] + [
            f"{key}={describe(item)}" for key, item in sorted(value.keywords.items())
        ]
        return f"{func}({', '.join(arguments)})"
    if isinstance(value, Mapping):
        return (
            "{" + ", ".join(f"{key}: {describe(item)}" for key, item in sorted(value.items())) + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(describe(item) for item in value) + "]"
    if isinstance(value, LabelTaxonomy):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    raise TypeError(f"{type(value).__name__} has no canonical text for the anchor cache key.")


class TrainingSplit:
    """Box supervised training sources of a datamodule, the data every recipe reads."""

    def __init__(self, sources: Sequence[DatasetSource], splitter: SplitterInterface) -> None:
        """Initialize the split.

        Args:
            sources: Training sources whose dataset is box supervised.
            splitter: Splitter assigning the records of a database to its splits.

        Raises:
            ValueError: If there is no source, or the dataset of a source builds no tasks.
        """
        if not sources:
            raise ValueError("The training split has no box supervised source to derive anchors.")
        untasked = [
            source.database.version
            for source in sources
            if source.train_dataset is None or not source.train_dataset.task_builders
        ]
        if untasked:
            raise ValueError(
                f"The training datasets over {untasked} build no dataset tasks, the anchors of "
                "the model read them."
            )
        self.sources = tuple(sources)
        self.splitter = splitter
        self._records: list[pl.DataFrame] | None = None

    def records(self) -> list[pl.DataFrame]:
        """Training records of every source, read from the record tables on the first call."""
        if self._records is None:
            # Sources over one database share its records
            databases = {source.database.database_hash: source.database for source in self.sources}
            train = {
                database_hash: split_database_records(database, self.splitter)[SplitType.TRAIN]
                for database_hash, database in databases.items()
            }
            self._records = [train[source.database.database_hash] for source in self.sources]
        return self._records

    def build_task(self, index: int, task_type: TaskType) -> BaseDatasetTask:
        """Build one dataset task of a source over its training records.

        Args:
            index: Index of the source.
            task_type: Task to build.

        Returns:
            The task over the training records of the source.
        """
        dataset = self.sources[index].train_dataset
        return self.task_builder(index, task_type)(
            database_root_path=str(dataset.database_root_path),
            dataset_records_dataframe=self.records()[index],
        )

    def task_builder(self, index: int, task_type: TaskType) -> Callable[..., BaseDatasetTask]:
        """Builder of one dataset task of a source.

        Args:
            index: Index of the source.
            task_type: Task asked for.

        Returns:
            The task builder of the training dataset of the source.

        Raises:
            ValueError: If the training dataset builds no such task.
        """
        source = self.sources[index]
        if task_type not in source.train_dataset.task_builders:
            raise ValueError(
                f"The training dataset over {source.database.version} builds no "
                f"{task_type.value} task, the anchors of the model read it."
            )
        return source.train_dataset.task_builders[task_type]

    def fingerprint(self, task_types: Sequence[TaskType]) -> str:
        """Identity of the data some tasks read: every database and the settings of its tasks.

        Args:
            task_types: Tasks a recipe reads.

        Returns:
            Canonical text of the training data.
        """
        return "; ".join(
            source.database.database_hash
            + ": "
            + ", ".join(
                f"{task_type}={describe(self.task_builder(index, task_type))}"
                for task_type in task_types
            )
            for index, source in enumerate(self.sources)
        )
