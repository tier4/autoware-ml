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

from collections.abc import Sequence
from typing import NamedTuple

from autoware_ml.datamodule.base_dataset import BaseDataset
from autoware_ml.datamodule.sources import DatasetSource, split_database_records
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType
from autoware_ml.types.tasks import TaskType


class TrainingDataset(NamedTuple):
    """The training dataset of a source, bound to the training records of its database.

    Attributes:
        source: The source.
        dataset: Its training dataset, bound like the datamodule binds it.
    """

    source: DatasetSource
    dataset: BaseDataset


class TrainingSplit:
    """Training sources of a datamodule, the data every recipe reads."""

    def __init__(self, sources: Sequence[DatasetSource], splitter: SplitterInterface) -> None:
        """Initialize the split.

        Args:
            sources: Sources serving the training split.
            splitter: Splitter assigning the records of a database to its splits.
        """
        self.sources = tuple(sources)
        self.splitter = splitter
        self._datasets: list[TrainingDataset] | None = None

    def datasets(self, task_types: Sequence[TaskType]) -> list[TrainingDataset]:
        """Training datasets that read their records through every given task.

        The datasets are bound to the training records of their databases on the first call,
        through the path the datamodule binds them on, so the tasks read the records training
        reads.

        Args:
            task_types: Tasks a recipe reads.

        Returns:
            The bound datasets building every given task, in source order.

        Raises:
            ValueError: If no training dataset builds every given task.
        """
        if self._datasets is None:
            # Sources over one database share its records
            databases = {source.database.database_hash: source.database for source in self.sources}
            records = {
                database_hash: split_database_records(database, self.splitter)[SplitType.TRAIN]
                for database_hash, database in databases.items()
            }
            self._datasets = [
                TrainingDataset(
                    source, source.train_dataset.bind(records[source.database.database_hash])
                )
                for source in self.sources
            ]
        selected = [
            training
            for training in self._datasets
            if all(task_type in training.dataset.dataset_tasks for task_type in task_types)
        ]
        if not selected:
            raise ValueError(
                f"No training dataset builds the {[str(task) for task in task_types]} tasks the "
                "anchors of the model read."
            )
        return selected

    def fingerprint(self, task_types: Sequence[TaskType]) -> str:
        """Identity of the data some tasks read: every database and the settings of its tasks.

        Args:
            task_types: Tasks a recipe reads.

        Returns:
            Canonical text of the training data.
        """
        return "; ".join(
            training.source.database.database_hash
            + ": "
            + ", ".join(
                f"{task_type}={training.dataset.dataset_tasks[task_type].hash_repr()}"
                for task_type in task_types
            )
            for training in self.datasets(task_types)
        )
