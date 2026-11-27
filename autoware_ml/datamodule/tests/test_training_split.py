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

"""Unit tests for the training split the anchor recipes read."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import polars as pl

from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.types.dataset import SplitType
from autoware_ml.types.tasks import TaskType


class _Task:
    """A dataset task stand-in with fixed settings."""

    def __init__(self, settings: str) -> None:
        self.settings = settings

    def hash_repr(self) -> str:
        return f"_Task({self.settings})"


class _Dataset:
    """A training dataset stand-in that builds its tasks when bound, like T4Dataset."""

    def __init__(self, tasks: Mapping[TaskType, str]) -> None:
        self.tasks = tasks
        self.dataset_tasks: Mapping[TaskType, _Task] = MappingProxyType({})
        self.bound_records: pl.DataFrame | None = None

    def bind(self, records: pl.DataFrame) -> _Dataset:
        bound = _Dataset(self.tasks)
        bound.bound_records = records
        bound.dataset_tasks = MappingProxyType(
            {task_type: _Task(settings) for task_type, settings in self.tasks.items()}
        )
        return bound


class _Splitter:
    """Every record of a database is a training record."""

    def split_by_polars_dataframe(
        self, dataset_records_dataframe: pl.DataFrame, scenarios: object
    ) -> dict[SplitType, pl.DataFrame]:
        return {SplitType.TRAIN: dataset_records_dataframe}


def _source(database_hash: str, tasks: Mapping[TaskType, str]) -> SimpleNamespace:
    """A source over one database with one training record."""
    database = SimpleNamespace(
        version=database_hash,
        database_hash=database_hash,
        scenarios={},
        load_polars_scenario_dataframe=lambda: pl.DataFrame({"record": [database_hash]}),
    )
    return SimpleNamespace(database=database, train_dataset=_Dataset(tasks))


class TestTrainingSplit(unittest.TestCase):
    def test_the_datasets_are_bound_to_their_training_records(self) -> None:
        """
        Input: one source over one database.
        Expected: the recipe reads the dataset bound like the datamodule binds it.
        """
        split = TrainingSplit([_source("hash-a", {TaskType.DETECTION3D: "a"})], _Splitter())

        (training,) = split.datasets((TaskType.DETECTION3D,))

        self.assertEqual(training.dataset.bound_records["record"].to_list(), ["hash-a"])

    def test_a_recipe_reads_the_sources_building_its_tasks(self) -> None:
        """
        Input: a detection source and a motion source.
        Expected: each recipe reads the sources building every task it asks for.
        """
        split = TrainingSplit(
            [
                _source("det", {TaskType.DETECTION3D: "a"}),
                _source("motion", {TaskType.DETECTION3D: "a", TaskType.MOTION: "m"}),
            ],
            _Splitter(),
        )

        def names(task_types: tuple[TaskType, ...]) -> list[str]:
            return [training.source.database.version for training in split.datasets(task_types)]

        self.assertEqual(names((TaskType.DETECTION3D,)), ["det", "motion"])
        self.assertEqual(names((TaskType.MOTION,)), ["motion"])

    def test_tasks_no_source_builds_are_refused(self) -> None:
        """
        Input: a split whose only source builds the detection task.
        Expected: a recipe reading the segmentation task finds no data and says so.
        """
        split = TrainingSplit([_source("hash-a", {TaskType.DETECTION3D: "a"})], _Splitter())

        with self.assertRaisesRegex(ValueError, "No training dataset builds"):
            split.fingerprint((TaskType.SEGMENTATION3D,))

    def test_the_fingerprint_names_every_database_and_task_setting(self) -> None:
        """
        Input: two sources over two databases, then the second with another task setting.
        Expected: the text names both database hashes and the settings of their tasks, and
          changes with a task setting.
        """
        tasks = (TaskType.DETECTION3D,)

        def fingerprint(setting: str) -> str:
            sources = [
                _source("hash-a", {TaskType.DETECTION3D: "a"}),
                _source("hash-b", {TaskType.DETECTION3D: setting}),
            ]
            return TrainingSplit(sources, _Splitter()).fingerprint(tasks)

        text = fingerprint("b")

        self.assertIn("hash-a", text)
        self.assertIn("hash-b: detection3d=_Task(b)", text)
        self.assertNotEqual(text, fingerprint("c"))
