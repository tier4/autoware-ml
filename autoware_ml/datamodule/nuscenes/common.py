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

"""Shared NuScenes path helpers.

This module centralizes path resolution and dataset-specific loading utilities
shared by NuScenes task adapters.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np


def resolve_lidar_path(data_root: str, relative_path: str) -> str:
    """Resolve a NuScenes lidar path from an annotation entry.

    Args:
        data_root: Dataset root directory.
        relative_path: Relative or absolute lidar path stored in the annotations.

    Returns:
        Absolute filesystem path to the lidar file.
    """
    if os.path.isabs(relative_path):
        return relative_path
    if os.sep not in relative_path:
        return os.path.join(data_root, "samples", "LIDAR_TOP", relative_path)
    return os.path.join(data_root, relative_path)


def lidar_to_map(sample: Mapping[str, Any], ego_pose: np.ndarray | None = None) -> np.ndarray:
    """Transform from the lidar frame of a sample to the map frame.

    NuScenes keeps its points and boxes in the LIDAR_TOP frame, not in base_link, so the
    ego pose alone would place them a sensor mounting off the map. The evaluation filters
    read one ``ego2global`` meaning sensor frame to map, so the mounting is composed in
    here.

    Args:
        sample: NuScenes annotation record of the frame.
        ego_pose: Ego pose to compose with, read from ``sample`` when omitted.

    Returns:
        The 4x4 lidar-to-map transform.
    """
    pose = (
        np.asarray(sample["ego2global"], dtype=np.float64)
        if ego_pose is None
        else np.asarray(ego_pose, dtype=np.float64)
    )
    lidar2ego = np.asarray(sample["lidar_points"]["lidar2ego"], dtype=np.float64)
    return pose @ lidar2ego
