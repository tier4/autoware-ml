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

import numpy as np
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


def build_sweep_entries(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert stored ``lidar_sweeps`` metadata into loader-ready sweep entries.

    Each entry carries the sweep point-cloud path, its timestamp, and the
    rigid transform from the sweep lidar frame into the key lidar frame,
    composed from the stored ego poses and lidar extrinsics.

    Args:
        sample: Raw sample dictionary with ``lidar_points``, ``ego2global``,
            and ``lidar_sweeps`` metadata.

    Returns:
        Sweep dictionaries consumed by ``LoadPointsFromMultiSweeps``.

    Raises:
        KeyError: If a sweep is missing the pose or path metadata required to
            express it in the key lidar frame.
    """
    lidar_sweeps = sample.get("lidar_sweeps")
    if not lidar_sweeps:
        return []

    key_lidar2ego = np.asarray(sample["lidar_points"]["lidar2ego"], dtype=np.float64)
    key_ego2global = np.asarray(sample["ego2global"], dtype=np.float64)
    global2key_lidar = np.linalg.inv(key_ego2global @ key_lidar2ego)

    entries = []
    for sweep in lidar_sweeps:
        sweep_lidar2ego = np.asarray(sweep["lidar_points"]["lidar2ego"], dtype=np.float64)
        sweep_ego2global = np.asarray(sweep["ego2global"], dtype=np.float64)
        sweep2key_lidar = global2key_lidar @ sweep_ego2global @ sweep_lidar2ego
        entries.append(
            {
                "lidar_path": sweep["lidar_points"]["lidar_path"],
                "timestamp": sweep["timestamp"],
                "sensor2lidar_rotation": sweep2key_lidar[:3, :3].astype(np.float32),
                "sensor2lidar_translation": sweep2key_lidar[:3, 3].astype(np.float32),
            }
        )
    return entries


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
    if "sweeps" not in normalized:
        normalized["sweeps"] = build_sweep_entries(normalized)
    return normalized


def load_detection_data_infos(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Load normalized detection samples from an annotation payload.

    Args:
        data: Deserialized annotation payload.

    Returns:
        Normalized list of detection samples.
    """
    return [normalize_detection_sample(sample) for sample in data["data_list"]]


def build_label_to_category(metainfo: Mapping[str, Any]) -> dict[int, str]:
    """Build a ``{label_index: category_name}`` map from annotation metainfo.

    Supports both annotation schemas:
      - ``categories`` (a ``{name: index}`` mapping, e.g. the nuScenes converter), and
      - ``classes`` (an ordered list, e.g. the T4 converter),

    so the loaded ``bbox_label_3d`` indices can be decoded to class names.

    Args:
        metainfo: ``metainfo`` block from a deserialized annotation payload.

    Returns:
        Mapping from stored label index to category name.
    """
    categories = metainfo.get("categories")
    if categories:
        return {int(index): str(name) for name, index in categories.items()}
    classes = metainfo.get("classes")
    if classes:
        return {label: str(category) for label, category in enumerate(classes)}
    raise ValueError("Annotation file metainfo must define 'categories' or 'classes'.")


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
