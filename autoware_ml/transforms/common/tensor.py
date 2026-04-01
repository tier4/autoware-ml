"""Tensor conversion and axis permutation transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


class ToTensor(BaseTransform):
    """Convert nested NumPy and scalar values into PyTorch tensors."""

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert all values in the input dictionary to tensor-friendly types."""
        return {key: self._convert(value) for key, value in input_dict.items()}

    def _convert(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, str):
            return data
        if isinstance(data, int):
            return torch.LongTensor([data])
        if isinstance(data, float):
            return torch.FloatTensor([data])
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        if isinstance(data, Mapping):
            return {sub_key: self._convert(item) for sub_key, item in data.items()}
        if isinstance(data, Sequence):
            return [self._convert(item) for item in data]
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")
