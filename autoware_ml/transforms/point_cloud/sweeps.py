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


SWEEP_SELECTIONS = frozenset({"nearest", "random"})


class LoadPointsFromMultiSweeps(BaseTransform):
    """Append historical sweep points to the current point-cloud frame.

    When ``time_dim`` is set, the transform overwrites that raw feature column
    with the per-point time lag relative to the current frame (``0`` for the
    current frame, ``key_timestamp - sweep_timestamp`` in seconds for sweeps)
    before applying ``use_dim``. In that mode the transform must be the point
    loader for the sample so the raw column layout is known.

    The current frame is always the leading block of the output and its size is
    exposed as ``num_current_points`` so label pipelines can pad the unlabeled
    sweep points.

    Which stored sweeps are appended is declared, not inferred. ``time_lag_range``
    bounds the age of an eligible sweep in seconds, and ``sweep_selection`` picks
    among the eligible ones: ``"nearest"`` takes the most recent, which is what
    evaluation and deployment see, and ``"random"`` samples them, which varies the
    temporal baseline during training so the network has to read the time lag
    instead of assuming a fixed frame interval. Sweeps outside the window count as
    unavailable, exactly like the missing sweep of a scene's first frame.

    """

    _optional_keys = ["points"]

    def __init__(
        self,
        *,
        sweeps_num: int,
        load_dim: int = 5,
        use_dim: Sequence[int] | None = None,
        time_dim: int | None = None,
        sweep_selection: str,
        time_lag_range: Sequence[float],
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        close_radius: float = 1.0,
    ) -> None:
        """Initialize the LoadPointsFromMultiSweeps transform.

        Args:
            sweeps_num: Number of sweeps included in the output including the
                current frame.
            load_dim: Number of features stored per point in sweep files.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
            time_dim: Optional raw feature column overwritten with the time lag
                relative to the current frame before ``use_dim`` selection.
            sweep_selection: How to pick the appended sweeps among the eligible
                entries: ``"nearest"`` takes the most recent ones, ``"random"``
                samples them uniformly without replacement.
            time_lag_range: Inclusive ``[min, max]`` age in seconds an eligible
                sweep may have, with ``0 < min < max``. Entries outside it are
                treated as unavailable.
            pad_empty_sweeps: Whether to repeat the current frame when no sweeps exist.
                With ``time_dim`` set the copies carry the minimum time lag, so
                current-frame selections never count them.
            remove_close: Whether to drop sweep points close to the origin.
            close_radius: Half-width in meters of the removed region when
                ``remove_close`` is enabled.
        """
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.use_dim = tuple(use_dim) if use_dim is not None else tuple(range(min(load_dim, 4)))
        if time_dim is not None and not 0 <= time_dim < load_dim:
            raise ValueError(f"time_dim must be within [0, {load_dim}), got {time_dim}.")
        self.time_dim = time_dim
        if sweep_selection not in SWEEP_SELECTIONS:
            raise ValueError(
                f"sweep_selection must be one of {sorted(SWEEP_SELECTIONS)}, "
                f"got {sweep_selection!r}."
            )
        if len(time_lag_range) != 2:
            raise ValueError(f"time_lag_range must contain [min, max], got {time_lag_range}.")
        min_time_lag, max_time_lag = (float(value) for value in time_lag_range)
        if min_time_lag <= 0.0 or min_time_lag >= max_time_lag:
            raise ValueError(
                f"Expected 0 < min time lag < max time lag, got {time_lag_range}. The current "
                "frame owns lag 0, a zero minimum would let a zero-lag sweep masquerade as it."
            )
        self.sweep_selection = sweep_selection
        self.min_time_lag = min_time_lag
        self.max_time_lag = max_time_lag
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.close_radius = close_radius

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Load the current-frame point cloud when it is not present yet."""
        if "points" in input_dict:
            return
        if "lidar_path" not in input_dict:
            raise KeyError("LoadPointsFromMultiSweeps requires 'points' or 'lidar_path'")
        if "idx_begin" in input_dict or "length" in input_dict:
            raise ValueError(
                "LoadPointsFromMultiSweeps loads whole frames and does not support the "
                "per-sensor 'idx_begin'/'length' slicing of the sample."
            )

        load_dim = int(input_dict.get("num_pts_feats", self.load_dim))
        points = np.fromfile(input_dict["lidar_path"], dtype=np.float32).reshape(-1, load_dim)
        if self.time_dim is None:
            points = points[:, self.use_dim]
        input_dict["points"] = points.astype(np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Append sweep points to the current frame."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        if self.time_dim is not None:
            if points.shape[1] != self.load_dim:
                raise ValueError(
                    "LoadPointsFromMultiSweeps with time_dim must load the raw point layout: "
                    f"expected {self.load_dim} columns, got {points.shape[1]}. The transform "
                    "must be the point loader for the sample."
                )
            points = points.copy()
            points[:, self.time_dim] = 0.0

        input_dict["num_current_points"] = points.shape[0]
        sweep_entries = list(input_dict.get("sweeps", []))
        if not sweep_entries:
            if self.pad_empty_sweeps and self.sweeps_num > 1:
                padding = np.tile(points, (self.sweeps_num - 1, 1))
                if self.time_dim is not None:
                    # Padded copies stand in for sweeps, so they carry the youngest
                    # admissible lag and never masquerade as current-frame points.
                    padding[:, self.time_dim] = self.min_time_lag
                points = np.concatenate([points, padding], axis=0)
            input_dict["points"] = self._select_dims(points)
            return input_dict

        needed = max(0, self.sweeps_num - 1)
        key_timestamp = self._key_timestamp(input_dict)
        selected_sweeps = self._select_sweeps(sweep_entries, needed, key_timestamp)
        sweep_points = [points]
        for time_lag, sweep in selected_sweeps:
            sweep_array = self._load_sweep_points(sweep).copy()
            if self.time_dim is not None:
                sweep_array[:, self.time_dim] = time_lag
            if self.remove_close:
                sweep_array = self._remove_close_points(sweep_array)
            rotation = np.asarray(sweep.get("sensor2lidar_rotation", np.eye(3)), dtype=np.float32)
            translation = np.asarray(
                sweep.get("sensor2lidar_translation", np.zeros(3)), dtype=np.float32
            )
            sweep_array[:, :3] = sweep_array[:, :3] @ rotation.T + translation
            sweep_points.append(sweep_array)

        input_dict["points"] = self._select_dims(np.concatenate(sweep_points, axis=0))
        return input_dict

    @staticmethod
    def _key_timestamp(input_dict: Mapping[str, Any]) -> float:
        """Return the current frame's capture time, which every sweep age is measured against."""
        if input_dict.get("timestamp") is None:
            raise KeyError("LoadPointsFromMultiSweeps requires 'timestamp' to age the sweeps.")
        return float(input_dict["timestamp"])

    def _select_sweeps(
        self, sweep_entries: Sequence[Mapping[str, Any]], needed: int, key_timestamp: float
    ) -> list[tuple[float, Mapping[str, Any]]]:
        """Return the sweeps to append, newest first, paired with their age in seconds.

        Entries outside ``time_lag_range`` are unavailable, so a scene whose previous frames were
        dropped yields fewer sweeps rather than a stale one.
        """
        if needed == 0:
            return []
        eligible = []
        for sweep in sweep_entries:
            if sweep.get("timestamp") is None:
                raise KeyError("LoadPointsFromMultiSweeps requires sweep 'timestamp'.")
            time_lag = key_timestamp - float(sweep["timestamp"])
            if self.min_time_lag <= time_lag <= self.max_time_lag:
                eligible.append((time_lag, sweep))
        eligible.sort(key=lambda entry: entry[0])
        if self.sweep_selection == "random" and len(eligible) > needed:
            indices = np.random.choice(len(eligible), needed, replace=False)
            return sorted((eligible[index] for index in indices), key=lambda entry: entry[0])
        return eligible[:needed]

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
        if self.time_dim is None:
            points = points[:, self.use_dim]
        return points

    def _remove_close_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Remove points close to the origin in the xy plane.

        The removed region is the axis-aligned box |x|, |y| < close_radius
        """
        close = (np.abs(points[:, 0]) < self.close_radius) & (
            np.abs(points[:, 1]) < self.close_radius
        )
        return points[~close]
