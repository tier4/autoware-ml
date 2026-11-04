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


"""Anchors given up front, for the tests of the heads."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch.nn as nn

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.models.anchored import DataAnchored


class FixedAnchorRecipe(AnchorRecipe):
    """Recipe that returns the arrays it was given."""

    name = "fixed"
    task_types = ()

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Initialize the recipe.

        Args:
            arrays: Named anchor arrays.
        """
        super().__init__(cache_path="fixed_anchors")
        self.arrays = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}

    def hash_repr(self) -> str:
        """Canonical text of the recipe."""
        return f"{type(self).__name__}({sorted(self.arrays)})"

    def build(self, split: TrainingSplit) -> dict[str, np.ndarray]:
        """Return the given arrays."""
        return dict(self.arrays)


def fill_anchors(module: nn.Module) -> nn.Module:
    """Fill every module holding fixed anchors, as training does from the anchor cache.

    Args:
        module: Head or model whose submodules use fixed anchor recipes.

    Returns:
        The same module.
    """
    for child in module.modules():
        if isinstance(child, DataAnchored):
            child.initialize_anchors(child.anchor_recipe.arrays)
    return module
