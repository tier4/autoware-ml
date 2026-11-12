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

"""Per-frame evaluation metadata derived from the dataset infos.

The region and collision evaluation filters need two things the raw model inputs
do not carry: the frame's ego pose in the map frame (attached directly as
``ego2global``) and a scene identifier the lanelet map provider can resolve to
the scene's ``map/lanelet2_map.osm``. The scene identifier is derived here; the
datasets attach both at metadata time and the eval-output builders pass them
through per frame.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def scene_dir_fragment(lidar_relative_path: str) -> str:
    """Scene directory fragment ``<db>/<uuid>/<version>`` from a stored lidar path.

    This is the ``scene_token`` convention the metrics use: unlike the opaque
    annotation token, it lets :class:`T4LaneletMapResolver` locate the scene's
    lanelet map directly under the database root. The records store every path
    relative to that root, so the fragment is the first three components of the
    stored path.

    Args:
        lidar_relative_path: Lidar file path of the record, relative to the database root.

    Returns:
        The ``<db>/<uuid>/<version>`` fragment below the root.

    Raises:
        ValueError: If the path is absolute or does not start at a scene directory.
    """
    path = PurePosixPath(lidar_relative_path)
    if path.is_absolute():
        raise ValueError(
            f"Cannot derive a scene directory from lidar path {lidar_relative_path!r}: the "
            "records store their paths relative to the database root."
        )
    parts = path.parts
    if len(parts) < 4:
        raise ValueError(
            f"Cannot derive a scene directory from lidar path {lidar_relative_path!r}; expected "
            "'<db>/<scene_uuid>/<version>/...'."
        )
    return "/".join(parts[:3])
