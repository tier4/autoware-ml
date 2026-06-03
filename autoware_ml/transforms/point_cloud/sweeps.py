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

"""Point-cloud sweep loading transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


class LoadPointsFromMultiSweeps(BaseTransform):
    """Append historical sweep points to the current point-cloud frame."""

    _optional_keys = ["points"]

    def __init__(
        self,
        sweeps_num: int,
        load_dim: int = 5,
        use_dim: Sequence[int] | None = None,
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        close_radius: float = 1.0,
    ) -> None:
        """Initialize the multi-sweep point loader."""
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.use_dim = tuple(use_dim) if use_dim is not None else tuple(range(min(load_dim, 4)))
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.close_radius = close_radius

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Load the current-frame point cloud when it is not present yet."""
        if "points" in input_dict:
            return
        if "lidar_path" not in input_dict:
            raise KeyError("LoadPointsFromMultiSweeps requires 'points' or 'lidar_path'")

        load_dim = int(input_dict.get("num_pts_feats", self.load_dim))
        points = np.fromfile(input_dict["lidar_path"], dtype=np.float32).reshape(-1, load_dim)
        input_dict["points"] = points[:, self.use_dim].astype(np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Append sweep points to the current frame."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        sweep_entries = list(input_dict.get("sweeps", []))
        if not sweep_entries:
            if self.pad_empty_sweeps:
                input_dict["points"] = np.concatenate([points] * self.sweeps_num, axis=0)
            return input_dict

        selected_sweeps = sweep_entries[: max(0, self.sweeps_num - 1)]
        sweep_points = [points]
        for sweep in selected_sweeps:
            sweep_array = self._load_sweep_points(sweep)
            if self.remove_close:
                sweep_array = self._remove_close_points(sweep_array)
            rotation = np.asarray(sweep.get("sensor2lidar_rotation", np.eye(3)), dtype=np.float32)
            translation = np.asarray(
                sweep.get("sensor2lidar_translation", np.zeros(3)), dtype=np.float32
            )
            sweep_array = sweep_array.copy()
            sweep_array[:, :3] = sweep_array[:, :3] @ rotation.T + translation
            sweep_points.append(sweep_array)

        input_dict["points"] = np.concatenate(sweep_points, axis=0)
        return input_dict

    def _load_sweep_points(self, sweep: Mapping[str, Any]) -> npt.NDArray[np.float32]:
        """Load one sweep point cloud from memory or from disk."""
        if "points" in sweep:
            points = np.asarray(sweep["points"], dtype=np.float32)
        else:
            lidar_path = os.fspath(sweep["lidar_path"])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, self.load_dim)
        return points[:, self.use_dim]

    def _remove_close_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Remove points close to the origin in the xy plane."""
        radius = np.linalg.norm(points[:, :2], axis=1)
        return points[radius >= self.close_radius]
