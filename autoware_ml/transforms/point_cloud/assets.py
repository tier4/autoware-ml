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

"""Locate assets bundled with the package.

Per-platform assets (LiDAR masks, crop-box definitions) and per-sensor calibration each live in
their own subdirectory of ``autoware_ml/configs/assets``. Nothing here knows which platforms exist:
callers name the directory they want.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

ASSET_ROOT = Path(str(resources.files("autoware_ml.configs").joinpath("assets")))


def resolve_asset_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve an asset path, treating relative paths as bundled.

    Args:
        path: Absolute path, or a path relative to ``root``.
        root: Directory relative paths resolve under. Defaults to the bundled asset root.

    Returns:
        The resolved path. It is not checked for existence.
    """
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (ASSET_ROOT if root is None else root) / resolved
