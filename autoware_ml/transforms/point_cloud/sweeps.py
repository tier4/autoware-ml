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

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.ego_motion import PRE_CORRECTION_POINTS_KEY
from autoware_ml.transforms.point_cloud.lidar_sources import (
    SAMPLE_DATA_TABLE,
    resolve_sources_info,
)


class LoadPointsFromMultiSweeps(BaseTransform):
    """Append historical sweep points to the current point-cloud frame.

    When ``time_dim`` is set, the transform overwrites that raw feature column
    with the per-point time lag relative to the current frame (``0`` for the
    current frame, ``key_timestamp - sweep_timestamp`` in seconds for sweeps)
    before applying ``use_dim``. In that mode the transform must be the point
    loader for the sample so the raw column layout is known.

    ``sweep_transforms`` reproduces per-scan vehicle-side filtering on the sweeps. It has to happen
    here, not later in the pipeline, because this transform destroys what such filters need: each
    sweep is rigidly moved into the current frame, and with ``time_dim`` set the per-point
    acquisition timestamps are overwritten with a single per-sweep lag. Afterwards nothing
    downstream can tell which rows came from which sweep, which LiDAR within it, or when.
    """

    _optional_keys = ["points"]

    def __init__(
        self,
        *,
        sweeps_num: int,
        load_dim: int = 5,
        use_dim: Sequence[int] | None = None,
        time_dim: int | None = None,
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        close_radius: float = 1.0,
        sweep_transforms: Any = None,
    ) -> None:
        """Initialize the LoadPointsFromMultiSweeps transform.

        Args:
            sweeps_num: Number of sweeps included in the output including the
                current frame.
            load_dim: Number of features stored per point in sweep files.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
            time_dim: Optional raw feature column overwritten with the time lag
                relative to the current frame before ``use_dim`` selection.
            pad_empty_sweeps: Whether to repeat the current frame when no sweeps exist.
            remove_close: Whether to drop sweep points close to the origin.
            close_radius: Radius in meters used when ``remove_close`` is enabled.
            sweep_transforms: Pipeline applied to each historical sweep on its own, in its own frame
                and with its own timestamps, before it is moved into the current frame. Use it for
                filtering the vehicle applied per scan ahead of concatenation. The current frame is
                *not* passed through it -- it is already in the sample, where the pipeline can filter
                it alongside its own per-point annotations.
        """
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.use_dim = tuple(use_dim) if use_dim is not None else tuple(range(min(load_dim, 4)))
        if time_dim is not None and not 0 <= time_dim < load_dim:
            raise ValueError(f"time_dim must be within [0, {load_dim}), got {time_dim}.")
        self.time_dim = time_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.close_radius = close_radius
        self.sweep_transforms = sweep_transforms

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Load the current-frame point cloud when it is not present yet."""
        if "points" in input_dict:
            return
        if "lidar_path" not in input_dict:
            raise KeyError("LoadPointsFromMultiSweeps requires 'points' or 'lidar_path'")

        load_dim = int(input_dict.get("num_pts_feats", self.load_dim))
        points = np.fromfile(input_dict["lidar_path"], dtype=np.float32).reshape(-1, load_dim)
        if self.time_dim is None:
            points = points[:, self.use_dim]
        input_dict["points"] = points.astype(np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Append sweep points to the current frame."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        key_timestamp = 0.0
        if self.time_dim is not None:
            if points.shape[1] != self.load_dim:
                raise ValueError(
                    "LoadPointsFromMultiSweeps with time_dim must load the raw point layout: "
                    f"expected {self.load_dim} columns, got {points.shape[1]}. The transform "
                    "must be the point loader for the sample."
                )
            if input_dict.get("timestamp") is None:
                raise KeyError("LoadPointsFromMultiSweeps with time_dim requires 'timestamp'.")
            key_timestamp = float(input_dict["timestamp"])
            points = points.copy()
            points[:, self.time_dim] = 0.0

        sweep_entries = list(input_dict.get("sweeps", []))
        if not sweep_entries:
            if self.pad_empty_sweeps:
                points = np.concatenate([points] * self.sweeps_num, axis=0)
            input_dict["points"] = self._select_dims(points)
            return input_dict

        selected_sweeps = sweep_entries[: max(0, self.sweeps_num - 1)]
        sweep_points = [points]
        for sweep in selected_sweeps:
            sweep_array = self._load_sweep_points(sweep).copy()
            if self.sweep_transforms is not None:
                # While the sweep is still where and when it was captured.
                sweep_array = self._transform_sweep(sweep_array, sweep, input_dict)
                if self.time_dim is None:
                    sweep_array = sweep_array[:, self.use_dim]
            if self.time_dim is not None:
                if sweep.get("timestamp") is None:
                    raise KeyError(
                        "LoadPointsFromMultiSweeps with time_dim requires sweep 'timestamp'."
                    )
                sweep_array[:, self.time_dim] = key_timestamp - float(sweep["timestamp"])
            if self.remove_close:
                sweep_array = self._remove_close_points(sweep_array)
            rotation = np.asarray(sweep.get("sensor2lidar_rotation", np.eye(3)), dtype=np.float32)
            translation = np.asarray(
                sweep.get("sensor2lidar_translation", np.zeros(3)), dtype=np.float32
            )
            sweep_array[:, :3] = sweep_array[:, :3] @ rotation.T + translation
            sweep_points.append(sweep_array)

        input_dict["points"] = self._select_dims(np.concatenate(sweep_points, axis=0))
        # Anything the current frame carried per point describes the current frame alone, and the
        # cloud is now longer than it. Pre-correction coordinates are the one such array in play;
        # left behind, they would ride along at the wrong length and be filtered as though aligned
        # on any sample where the two counts happened to coincide.
        input_dict.pop(PRE_CORRECTION_POINTS_KEY, None)
        return input_dict

    def _select_dims(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply ``use_dim`` selection deferred to the end in time-lag mode."""
        if self.time_dim is None:
            return points
        return points[:, self.use_dim]

    def _load_sweep_points(self, sweep: Mapping[str, Any]) -> npt.NDArray[np.float32]:
        """Load one sweep point cloud from memory or from disk."""
        if "points" in sweep:
            points = np.asarray(sweep["points"], dtype=np.float32)
        else:
            lidar_path = os.fspath(sweep["lidar_path"])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, self.load_dim)
        # Per-scan transforms index by raw feature column, so the selection waits until they have run.
        if self.time_dim is None and self.sweep_transforms is None:
            points = points[:, self.use_dim]
        return points

    def _transform_sweep(
        self,
        points: npt.NDArray[np.float32],
        sweep: Mapping[str, Any],
        input_dict: Mapping[str, Any],
    ) -> npt.NDArray[np.float32]:
        """Run ``sweep_transforms`` over one sweep as a sample in its own right.

        The sweep is given its own concat metadata and its own path, so filters resolve its LiDAR
        ranges, its reference time and its scene's ego poses rather than the current frame's. Sensor
        extrinsics come from the sample, being static calibration shared by every frame of a scene.

        Raises:
            KeyError: The sweep has no concat metadata, so its points cannot be attributed to the
                LiDARs that produced them.
        """
        sources_info = resolve_sources_info(sweep)
        if sources_info is None:
            raise KeyError(
                "sweep_transforms needs per-sweep concat metadata: none was given as "
                "'lidar_sources_info', none was named by 'lidar_pointcloud_source_path', and "
                f"{SAMPLE_DATA_TABLE} records no info_filename for "
                f"{sweep.get('lidar_path')!r}."
            )
        sub_sample = {
            "points": points,
            "lidar_path": sweep.get("lidar_path"),
            "lidar_sources": input_dict.get("lidar_sources"),
            "lidar_sources_info": sources_info,
            "timestamp": sweep.get("timestamp"),
        }
        return np.asarray(self.sweep_transforms(sub_sample)["points"], dtype=np.float32)

    def _remove_close_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Remove points close to the origin in the xy plane."""
        radius = np.linalg.norm(points[:, :2], axis=1)
        return points[radius >= self.close_radius]
