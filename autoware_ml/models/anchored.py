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


"""Modules whose anchors come from the training split through the anchor cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float32

from autoware_ml.anchors.base import AnchorRecipe


def anchor_tensor(
    arrays: Mapping[str, Float32[np.ndarray, "..."]], key: str, shape: Sequence[int]
) -> Float32[torch.Tensor, "..."]:
    """One anchor array of a recipe as a float tensor of the expected shape.

    Args:
        arrays: Named anchor arrays of the recipe.
        key: Name of the array.
        shape: Shape the module expects.

    Returns:
        The array as a float32 tensor.

    Raises:
        ValueError: If the array has another shape.
    """
    array = arrays[key]
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"Anchors {key} have the shape {array.shape}, the module expects {shape}.")
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _load_whole_anchor_state(
    module: DataAnchored, state_dict: Mapping[str, Any], prefix: str, *_: Any
) -> None:
    """Raise when a state dict holds only part of the anchors of a module and their flag.

    Args:
        module: Module being loaded.
        state_dict: State dict handed to ``load_state_dict``, after any key filtering.
        prefix: Key prefix of the module.

    Raises:
        ValueError: If some but not all of ``anchors_ready`` and the anchor tensors are given.
    """
    keys = [prefix + name for name in ("anchors_ready", *module.anchor_names)]
    missing = [key for key in keys if key not in state_dict]
    if 0 < len(missing) < len(keys):
        raise ValueError(
            f"The anchors and anchors_ready load together, the state dict misses {missing}. "
            "An anchor tensor skipped for its shape would stay unfilled behind a ready flag, and "
            "weights without the flag would have their anchors rebuilt."
        )


class DataAnchored(nn.Module, ABC):
    """Module whose anchors the training split gives, filled once from the anchor cache.

    The anchors are state of the module, so loaded weights carry them and skip the cache. A
    resumed run reads the cache and the checkpoint restore then overwrites the anchors, see
    :meth:`ModuleBaseModel.configure_model`. A state dict loads the anchors and their
    ``anchors_ready`` flag together or none of them.
    """

    # Names of the state tensors that load_anchors fills
    anchor_names: ClassVar[tuple[str, ...]]

    def __init__(self, anchor_recipe: AnchorRecipe) -> None:
        """Initialize the module.

        Args:
            anchor_recipe: Recipe of the anchors.
        """
        super().__init__()
        self.anchor_recipe = anchor_recipe
        self.register_buffer("anchors_ready", torch.tensor(False))
        self.register_load_state_dict_pre_hook(_load_whole_anchor_state)

    @abstractmethod
    def load_anchors(self, arrays: Mapping[str, Float32[np.ndarray, "..."]]) -> None:
        """Copy the anchor arrays of the recipe into the module.

        Args:
            arrays: Named anchor arrays of the recipe.
        """

    @torch.no_grad()
    def initialize_anchors(self, arrays: Mapping[str, Float32[np.ndarray, "..."]]) -> None:
        """Fill the anchors and mark them ready.

        Args:
            arrays: Named anchor arrays of the recipe.
        """
        self.load_anchors(arrays)
        self.anchors_ready.fill_(True)
