"""Feature construction transforms shared across tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from autoware_ml.transforms.base import BaseTransform


class BuildPointFeatures(BaseTransform):
    """Build one point feature matrix from existing per-point arrays."""

    def __init__(
        self,
        *,
        keys: Sequence[str],
        output_key: str = "feat",
    ) -> None:
        """Initialize the BuildPointFeatures transform.

        Args:
            keys: Input keys concatenated along the feature axis.
            output_key: Destination key for the built point feature matrix.
        """
        self.keys = list(keys)
        self.output_key = output_key
        self._required_keys = self.keys

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Concatenate configured per-point fields into ``output_key``."""
        values = [input_dict[key] for key in self.keys]
        first = values[0]
        if isinstance(first, np.ndarray):
            if not all(isinstance(value, np.ndarray) for value in values):
                raise TypeError(
                    f"{self.__class__.__name__} requires all feature fields to be numpy arrays."
                )
            input_dict[self.output_key] = np.concatenate(
                [value.astype(np.float32, copy=False) for value in values],
                axis=1,
            )
            return input_dict
        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(
                    f"{self.__class__.__name__} requires all feature fields to be tensors."
                )
            input_dict[self.output_key] = torch.cat([value.float() for value in values], dim=1)
            return input_dict
        raise TypeError(
            f"{self.__class__.__name__} expects numpy arrays or torch tensors, "
            f"got {type(first)!r} for key '{self.keys[0]}'."
        )
