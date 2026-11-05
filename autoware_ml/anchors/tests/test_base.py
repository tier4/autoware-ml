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


"""Tests for the anchor cache of the recipes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.types.tasks import TaskType


class _CountingRecipe(AnchorRecipe):
    """Recipe that counts its builds."""

    name = "counting"
    task_types = (TaskType.DETECTION3D,)

    def __init__(self, cache_path: str, seed: int = 0) -> None:
        super().__init__(cache_path)
        self.seed = seed
        self.builds = 0

    def hash_repr(self) -> str:
        return f"_CountingRecipe(seed={self.seed})"

    def build(self, split: TrainingSplit) -> dict[str, Float[np.ndarray, "..."]]:
        self.builds += 1
        return {"anchors": np.full((2, 3), self.seed, dtype=np.float64)}


class _Split:
    """Training split stand-in with a fixed data fingerprint."""

    def __init__(self, fingerprint: str, splitter: str = "splitter") -> None:
        self._fingerprint = fingerprint
        self.splitter = splitter

    def fingerprint(self, task_types: Any) -> str:
        return f"{self._fingerprint} {list(task_types)}"


class TestAnchorCache(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_the_anchors_are_built_once_and_read_back(self) -> None:
        """
        Input: a recipe prepared twice on the same split, then loaded.
        Expected: one build, a float32 cache file alone under the cache path with the permissions of a
        plainly written file, the arrays read back.
        """
        recipe = _CountingRecipe(str(self.tmp_path))
        split = _Split("db-a")

        recipe.prepare(split)
        recipe.prepare(split)
        arrays = recipe.load(split)

        self.assertEqual(recipe.builds, 1)
        self.assertEqual(list(self.tmp_path.iterdir()), [recipe.cache_file(split)])
        reference = self.tmp_path / "reference"
        reference.write_bytes(b"")
        self.assertEqual(recipe.cache_file(split).stat().st_mode, reference.stat().st_mode)
        self.assertEqual(arrays["anchors"].dtype, np.float32)
        np.testing.assert_array_equal(arrays["anchors"], np.zeros((2, 3)))

    def test_the_cache_key_follows_the_recipe_and_the_data(self) -> None:
        """
        Input: one recipe on two databases and a recipe with another seed.
        Expected: three different cache files.
        """
        files = {
            _CountingRecipe(str(self.tmp_path)).cache_file(_Split("db-a")),
            _CountingRecipe(str(self.tmp_path)).cache_file(_Split("db-b")),
            _CountingRecipe(str(self.tmp_path), seed=1).cache_file(_Split("db-a")),
        }
        self.assertEqual(len(files), 3)

    def test_the_cache_key_follows_the_splitter(self) -> None:
        """
        Input: one recipe on one database split by two splitters.
        Expected: two cache files, the splitters select other training records.
        """
        recipe = _CountingRecipe(str(self.tmp_path))

        self.assertNotEqual(
            recipe.cache_file(_Split("db-a", "FirstSplitter()")),
            recipe.cache_file(_Split("db-a", "SecondSplitter()")),
        )

    def test_loading_anchors_that_were_not_built_raises(self) -> None:
        """
        Input: a recipe loaded before any preparation.
        Expected: a FileNotFoundError.
        """
        with self.assertRaisesRegex(FileNotFoundError, "prepare_data"):
            _CountingRecipe(str(self.tmp_path)).load(_Split("db-a"))
