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

"""Helpers for point-cloud dataloaders.

This module implements common point-cloud dataset and collation helpers used by
segmentation and other point-based learning tasks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data.dataloader import default_collate


def point_collate_fn(batch: list[Mapping[str, Any]], mix_prob: float = 0.0) -> dict[str, Any]:
    """Collate variable-length point-cloud samples.

    The returned batch concatenates point-level tensors and converts per-sample
    offsets to cumulative offsets matching the concatenated-offset input layout.
    """

    collated = _collate(batch)
    if "offset" in collated and mix_prob > 0.0:
        if torch.rand(1).item() < mix_prob and collated["offset"].numel() > 1:
            collated["offset"] = torch.cat(
                [collated["offset"][1:-1:2], collated["offset"][-1:].clone()],
                dim=0,
            )
    return collated


def _collate(batch: Sequence[Any]) -> Any:
    """Recursively collate nested point-cloud sample structures.

    Args:
        batch: Sequence of values from a dataset batch.

    Returns:
        Collated batch object with matching structure.
    """
    if not batch:
        raise ValueError("Batch must not be empty.")

    first = batch[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(list(batch), dim=0)
    if isinstance(first, str):
        return list(batch)
    if isinstance(first, Mapping):
        output = {key: _collate([item[key] for item in batch]) for key in first}
        for key, value in output.items():
            if "offset" in key:
                output[key] = torch.cumsum(value, dim=0)
        return output
    if isinstance(first, Sequence):
        return [_collate(items) for items in zip(*batch)]
    return default_collate(batch)
