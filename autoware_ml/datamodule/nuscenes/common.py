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
