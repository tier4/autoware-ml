"""Input packing transforms shared across tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from autoware_ml.transforms.base import BaseTransform


class Collect(BaseTransform):
    """Gather selected fields and optionally concatenate feature tensors."""

    def __init__(
        self,
        keys: str | Sequence[str],
        offset_keys_dict: Mapping[str, str] | None = None,
        **kwargs: Sequence[str],
    ) -> None:
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.offset_keys = dict(offset_keys_dict or {"offset": "coord"})
        self.kwargs = kwargs
        self._required_keys = list(self.keys) + list(self.offset_keys.values())
        for feature_keys in kwargs.values():
            self._required_keys.extend(list(feature_keys))

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Collect and pack selected fields."""
        data: dict[str, Any] = {key: input_dict[key] for key in self.keys}
        for output_key, source_key in self.offset_keys.items():
            data[output_key] = torch.tensor([input_dict[source_key].shape[0]])
        for name, keys in self.kwargs.items():
            feature_name = name.replace("_keys", "")
            data[feature_name] = torch.cat([input_dict[key].float() for key in keys], dim=1)
        return data
