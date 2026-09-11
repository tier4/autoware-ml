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

"""Lidar sample helpers shared by every point-cloud datamodule.

The helpers resolve stored point-cloud paths against the dataset root and turn
the stored historical sweep metadata into loader-ready entries expressed in the
key lidar frame.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


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


def build_sweep_entries(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert stored ``lidar_sweeps`` metadata into loader-ready sweep entries.

    Each entry carries the sweep point-cloud path, its timestamp, and the
    rigid transform from the sweep lidar frame into the key lidar frame.

    The precomputed ``lidar2sensor`` matrix (the key-lidar → sweep-lidar
    transform including ego motion) is preferred when present. Recomposing the
    transform from the stored ego poses is only a fallback

    Args:
        sample: Raw sample dictionary with ``lidar_points``, ``ego2global``,
            and ``lidar_sweeps`` metadata.

    Returns:
        Sweep dictionaries consumed by ``LoadPointsFromMultiSweeps``.

    Raises:
        KeyError: If a sweep is missing the pose or path metadata required to
            express it in the key lidar frame.
    """
    lidar_sweeps = sample.get("lidar_sweeps", [])
    if not lidar_sweeps:
        return []

    global2key_lidar = None

    entries = []
    for sweep in lidar_sweeps:
        lidar2sensor = sweep["lidar_points"].get("lidar2sensor")
        if lidar2sensor is not None:
            sweep2key_lidar = np.linalg.inv(np.asarray(lidar2sensor, dtype=np.float64))
        else:
            if global2key_lidar is None:
                key_lidar2ego = np.asarray(sample["lidar_points"]["lidar2ego"], dtype=np.float64)
                key_ego2global = np.asarray(sample["ego2global"], dtype=np.float64)
                global2key_lidar = np.linalg.inv(key_ego2global @ key_lidar2ego)
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


def resolve_sweep_paths(
    sweeps: Sequence[Mapping[str, Any]], data_root: str
) -> list[dict[str, Any]]:
    """Resolve sweep ``lidar_path`` entries against the dataset root.

    Args:
        sweeps: Loader-ready sweep entries, see :func:`build_sweep_entries`.
        data_root: Dataset root directory.

    Returns:
        Sweep dictionaries with ``lidar_path`` normalized when present.
    """
    sweep_entries = []
    for sweep in sweeps:
        sweep_entry = dict(sweep)
        if "lidar_path" in sweep_entry:
            sweep_entry["lidar_path"] = resolve_data_path(data_root, sweep_entry["lidar_path"])
        sweep_entries.append(sweep_entry)
    return sweep_entries
