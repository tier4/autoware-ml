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

"""Helpers for 3D detection dataloaders.

This module provides reusable dataset and datamodule components for lidar-based
3D detection tasks across supported datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any

from torch.utils.data import DataLoader, Dataset

from autoware_ml.datamodule.samplers import DistributedWeightedRandomSampler


def resolve_data_path(data_root: str, path: str) -> str:
    """Resolve a stored annotation path relative to a dataset root.

    Absolute paths are returned unchanged. Paths already nested under
    ``data_root`` are returned without re-prefixing. Other paths are joined
    with ``data_root``.

    Args:
        data_root: Dataset root directory.
        path: Stored annotation path to normalize.

    Returns:
        Absolute path or root-relative path resolved against ``data_root``.
    """
    normalized_path = os.path.normpath(path)
    normalized_root = os.path.normpath(data_root)
    if os.path.isabs(normalized_path):
        return normalized_path
    if normalized_path == normalized_root or normalized_path.startswith(normalized_root + os.sep):
        return normalized_path
    return os.path.join(normalized_root, normalized_path)


def resolve_sweep_paths(sample: Mapping[str, Any], data_root: str) -> list[dict[str, Any]]:
    """Resolve sweep ``lidar_path`` entries against the dataset root.

    Args:
        sample: Detection sample containing optional ``sweeps`` metadata.
        data_root: Dataset root directory.

    Returns:
        Sweep dictionaries with ``lidar_path`` normalized when present.
    """
    sweep_entries = []
    for sweep in sample.get("sweeps", []):
        sweep_entry = dict(sweep)
        if "lidar_path" in sweep_entry:
            sweep_entry["lidar_path"] = resolve_data_path(data_root, sweep_entry["lidar_path"])
        sweep_entries.append(sweep_entry)
    return sweep_entries


def normalize_detection_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize one detection annotation entry to the framework schema.

    Args:
        sample: Raw sample dictionary loaded from an annotation file.

    Returns:
        Sample dictionary that follows the internal detection convention with a
        top-level ``lidar_path`` field and list-backed optional collections.
    """
    normalized = dict(sample)
    if "lidar_path" not in normalized:
        normalized["lidar_path"] = normalized["lidar_points"]["lidar_path"]
    normalized.setdefault("instances", [])
    normalized.setdefault("sweeps", [])
    return normalized


def load_detection_data_infos(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Load normalized detection samples from an annotation payload.

    Args:
        data: Deserialized annotation payload.

    Returns:
        Normalized list of detection samples.
    """
    return [normalize_detection_sample(sample) for sample in data["data_list"]]


def build_detection_dataloader(
    dataset: Dataset,
    dataloader_cfg: Any,
    *,
    is_train: bool,
    train_frame_sampling: Any,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> DataLoader:
    """Build a dataloader for one 3D detection dataset split.

    Training loaders use ``DistributedWeightedRandomSampler`` when repeat-factor
    frame sampling is configured. In that mode, shuffling is disabled because
    sample order is controlled by the sampler. Non-training loaders use the
    dataloader configuration directly.

    Args:
        dataset: Dataset for the requested split. Training datasets must expose
            ``frame_weights`` when repeat-factor frame sampling is enabled.
        dataloader_cfg: Split dataloader configuration. It must provide
            ``to_dataloader_kwargs()`` and ``drop_last``.
        is_train: Whether the requested split is the training split.
        train_frame_sampling: Repeat-factor frame sampling configuration.
            A non-``None`` value enables weighted sampling for training.
        collate_fn: Callable passed to ``DataLoader`` to merge samples into a
            batch dictionary.

    Returns:
        Detection dataloader for the requested split.
    """
    dataloader_kwargs = dataloader_cfg.to_dataloader_kwargs()
    if is_train and train_frame_sampling is not None:
        dataloader_kwargs["shuffle"] = False
        dataloader_kwargs["sampler"] = DistributedWeightedRandomSampler(
            dataset,
            dataset.frame_weights,
            drop_last=dataloader_cfg.drop_last,
        )
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)
