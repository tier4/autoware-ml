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

"""Private helpers shared by segmentation3d transforms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def project_range(
    points: npt.NDArray[np.float32], height: int, width: int, fov_up_rad: float, fov_down_rad: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Project 3D points into range-view row and column indices."""
    depth = np.linalg.norm(points[:, :3], ord=2, axis=1)
    depth = np.clip(depth, a_min=1e-6, a_max=None)
    yaw = -np.arctan2(points[:, 1], points[:, 0])
    pitch = np.arcsin(np.clip(points[:, 2] / depth, -1.0, 1.0))
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    proj_x = 0.5 * (yaw / np.pi + 1.0) * width
    proj_y = (1.0 - (pitch + abs(fov_down_rad)) / fov) * height

    proj_x = np.clip(np.floor(proj_x), 0, width - 1).astype(np.int64)
    proj_y = np.clip(np.floor(proj_y), 0, height - 1).astype(np.int64)
    return proj_y, proj_x
