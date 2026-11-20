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

import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace


from autoware_ml.datamodule.training_split import TrainingSplit, describe
from autoware_ml.types.tasks import TaskType


class TestTrainingSplit(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_task_settings_have_one_canonical_text(self) -> None:
        """
        Input: a task builder with nested settings given in two orders, and an object.
        Expected: the same text for both orders, a TypeError for the object.
        """
        first = partial(dict, steps=12, roi={"x": 1.0, "y": 2.0}, classes=["a", "b"])
        second = partial(dict, classes=["a", "b"], roi={"y": 2.0, "x": 1.0}, steps=12)

        self.assertEqual(describe(first), describe(second))
        with self.assertRaisesRegex(TypeError, "canonical"):
            describe(object())

    def test_positional_builder_arguments_are_part_of_the_text(self) -> None:
        """
        Input: two task builders that differ only in a positional argument.
        Expected: two texts, so the anchors of one are never served to the other.
        """
        self.assertEqual(describe(partial(dict, 1, steps=12)), "builtins.dict(1, steps=12)")
        self.assertNotEqual(
            describe(partial(dict, 1, steps=12)), describe(partial(dict, 2, steps=12))
        )

    def test_a_source_without_a_task_the_anchors_read_is_refused(self) -> None:
        """
        Input: a training source whose dataset builds the detection task only.
        Expected: a recipe reading another task names the source, not a KeyError.
        """
        source = SimpleNamespace(
            database=SimpleNamespace(version="db-a", database_hash="hash-a"),
            train_dataset=SimpleNamespace(
                database_root_path=self.tmp_path, task_builders={TaskType.DETECTION3D: dict}
            ),
        )
        split = TrainingSplit([source], splitter=None)

        with self.assertRaisesRegex(ValueError, "over db-a builds no segmentation3d task"):
            split.fingerprint((TaskType.SEGMENTATION3D,))

    def test_a_source_whose_dataset_builds_no_tasks_is_refused(self) -> None:
        """
        Input: a training source whose dataset reads its records without dataset tasks.
        Expected: the split names the source when it is built, not an AttributeError later.
        """
        source = SimpleNamespace(
            database=SimpleNamespace(version="db-a", database_hash="hash-a"),
            train_dataset=SimpleNamespace(database_root_path=self.tmp_path, task_builders={}),
        )

        with self.assertRaisesRegex(ValueError, r"over \['db-a'\] build no dataset tasks"):
            TrainingSplit([source], splitter=None)

    def test_a_split_without_sources_is_refused(self) -> None:
        """
        Input: a training split of no source.
        Expected: ValueError, no recipe can derive anchors from it.
        """
        with self.assertRaisesRegex(ValueError, "no box supervised source"):
            TrainingSplit([], splitter=None)

    def test_the_fingerprint_names_every_database_and_task_setting(self) -> None:
        """
        Input: two sources over two databases, then the second with another task setting.
        Expected: the text names both database hashes and the settings of their tasks, and
          changes with a task setting.
        """

        def source(database_hash: str, steps: int) -> SimpleNamespace:
            return SimpleNamespace(
                database=SimpleNamespace(version=database_hash, database_hash=database_hash),
                train_dataset=SimpleNamespace(
                    database_root_path=self.tmp_path,
                    task_builders={TaskType.DETECTION3D: partial(dict, steps=steps)},
                ),
            )

        tasks = (TaskType.DETECTION3D,)
        text = TrainingSplit([source("hash-a", 1), source("hash-b", 2)], None).fingerprint(tasks)
        changed = TrainingSplit([source("hash-a", 1), source("hash-b", 3)], None).fingerprint(tasks)

        self.assertIn("hash-a", text)
        self.assertIn("hash-b", text)
        self.assertIn(describe(partial(dict, steps=2)), text)
        self.assertNotEqual(text, changed)
