"""Tensor conversion and axis permutation transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from autoware_ml.transforms.base import BaseTransform


class PermuteAxes(BaseTransform):
    """Permute axes of arrays or tensors for selected keys."""

    def __init__(self, input_keys: list[str], axes: tuple[int, ...]):
        self.input_keys = input_keys
        self.axes = axes
        self._required_keys = list(input_keys)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Permute axes of specified keys."""
        for key in self.input_keys:
            data = input_dict[key]
            if len(data.shape) != len(self.axes):
                raise ValueError(
                    f"PermuteAxes: Number of dimensions {data.shape} does not match "
                    f"number of axes {self.axes} for key '{key}'"
                )
            if isinstance(data, torch.Tensor):
                input_dict[key] = data.permute(*self.axes)
            elif isinstance(data, np.ndarray):
                input_dict[key] = np.transpose(data, self.axes)
        return input_dict
