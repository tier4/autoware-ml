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

"""Anchors the heads derive from the training split, cached like the record tables."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
from jaxtyping import Float, Float32

from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.types.tasks import TaskType

logger = logging.getLogger(__name__)


class AnchorRecipe(ABC):
    """Anchors a head derives from the training split.

    The model builds missing anchors into the cache before training, keyed by the recipe and the
    training data, and fills its heads from there. A checkpoint carries the anchors of its heads.
    """

    name: ClassVar[str]
    task_types: ClassVar[tuple[TaskType, ...]]

    def __init__(self, cache_path: str) -> None:
        """Initialize the recipe.

        Args:
            cache_path: Directory of the cached anchors.
        """
        self.cache_path = Path(cache_path)

    @abstractmethod
    def hash_repr(self) -> str:
        """Canonical text of the recipe settings, part of the cache key."""

    @abstractmethod
    def build(self, split: TrainingSplit) -> dict[str, Float[np.ndarray, "..."]]:
        """Derive the anchors from the training split.

        Args:
            split: Training sources and their records.

        Returns:
            Named anchor arrays.
        """

    def cache_file(self, split: TrainingSplit) -> Path:
        """Cache file of the anchors of the recipe on the training split.

        Args:
            split: Training sources.

        Returns:
            ``<cache_path>/<name>_<hash>.npz``, the hash over the recipe, the splitter and the
            training data.
        """
        key = f"{self.hash_repr()} | {split.splitter} | {split.fingerprint(self.task_types)}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_path / f"{self.name}_{digest}.npz"

    def prepare(self, split: TrainingSplit) -> None:
        """Build the anchors into the cache unless they are there.

        Args:
            split: Training sources and their records.
        """
        path = self.cache_file(split)
        if path.exists():
            logger.info(f"Anchors {path} already exist, skip building them")
            return
        arrays = self.build(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Renamed into place, so a concurrent reader never sees a partial file
        float_arrays = {key: value.astype(np.float32) for key, value in arrays.items()}
        # A plain exclusive open keeps the umask permissions of the record tables
        partial_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.partial")
        with open(partial_path, "xb") as partial_file:
            np.savez(partial_file, **float_arrays)
        os.replace(partial_path, path)
        logger.info(f"Saved anchors {[value.shape for value in arrays.values()]} to {path}")

    def load(self, split: TrainingSplit) -> dict[str, Float32[np.ndarray, "..."]]:
        """Read the cached anchors.

        Args:
            split: Training sources.

        Returns:
            Named anchor arrays.

        Raises:
            FileNotFoundError: If the anchors were not built.
        """
        path = self.cache_file(split)
        if not path.exists():
            raise FileNotFoundError(f"No anchors at {path}, the model builds them in prepare_data.")
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
