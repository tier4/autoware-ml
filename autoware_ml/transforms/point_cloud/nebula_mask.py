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

"""Reproduce the Nebula driver's per-LiDAR downsample mask."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.ego_motion import pre_correction_points
from autoware_ml.transforms.point_cloud.lidar_sources import (
    SourceSlice,
    iter_pointcloud_sources,
    normalize_lidar_name,
)


@dataclass(frozen=True)
class LidarMask:
    """One LiDAR's dithered downsample mask, plus the calibration it is indexed against.

    Load these with
    :func:`~autoware_ml.transforms.point_cloud.nebula_mask_assets.load_lidar_masks`, or construct
    them directly.

    Attributes:
        keep: ``(channels, azimuth_bins)`` boolean grid; ``True`` where the driver keeps a return.
        elevation_rad: Per-channel elevation, used only to estimate the ring index when the sample
            does not carry one.
        azimuth_deg: Per-channel azimuth correction, used only when a caller opts into
            ``use_calibration_azimuth_offsets``.
    """

    keep: npt.NDArray[np.bool_]
    elevation_rad: npt.NDArray[np.float32]
    azimuth_deg: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.keep.ndim != 2:
            raise ValueError(f"keep must be 2-D (channels, azimuth_bins); got {self.keep.shape}")
        channels = self.keep.shape[0]
        if self.elevation_rad.shape[0] != channels or self.azimuth_deg.shape[0] != channels:
            raise ValueError(
                f"mask has {channels} channels but calibration has "
                f"{self.elevation_rad.shape[0]} elevation and {self.azimuth_deg.shape[0]} azimuth "
                "entries"
            )


class NebulaDownsampleMaskFilter(BaseTransform):
    """Approximate Nebula's decode-time downsample mask on loaded Cartesian points.

    The vehicle filter runs in packet/range-view space before ego-motion correction. This transform
    reconstructs a per-LiDAR azimuth/channel index from Cartesian points, applies the same dithered
    mask, and keeps the original point rows in their existing coordinate frame.

    Masks are supplied already loaded, so this transform performs no file access. See
    :func:`~autoware_ml.transforms.point_cloud.nebula_mask_assets.load_lidar_masks` for reading a
    bundled platform's set.

    Required keys:
        points: Raw point array of shape ``(N, D)``.

    Optional keys:
        lidar_sources: Mapping from LiDAR name to calibration/extrinsic metadata.
        lidar_sources_info: PointCloudMetainfo-like dictionary containing source slices.
        source_name: Name of the single source when the sample has already been sliced.
        pre_correction_points: Pre-ego-motion-correction coordinates to decide on, as attached by
            :class:`~autoware_ml.transforms.point_cloud.ego_motion.InvertEgoMotionCorrection`.

    Generated keys:
        points and all aligned per-point numpy arrays filtered to the kept rows.
        nebula_downsample_stats when ``return_stats`` is true.
    """

    _required_keys = ["points"]

    def __init__(
        self,
        *,
        lidar_masks: Mapping[str, LidarMask],
        use_calibration_azimuth_offsets: bool = False,
        channel_dim: int | None = 4,
        azimuth_start_deg: float = 0.0,
        azimuth_extent_deg: float = 360.0,
        return_stats: bool = False,
    ) -> None:
        """Initialize the Nebula mask-filter approximation.

        Args:
            lidar_masks: Mask per LiDAR, keyed by mounting position (``"front_upper"``) or full
                channel name (``"LIDAR_FRONT_UPPER"``).
            use_calibration_azimuth_offsets: Whether to subtract per-channel azimuth calibration
                offsets before sampling the mask column. Defaults to ``False``: the driver indexes
                the mask by raw azimuth, and subtracting the offsets moves points into the wrong
                column.
            channel_dim: Index of the per-point feature holding the ring/channel number. Set to
                ``None`` to always estimate the ring from elevation instead.
            azimuth_start_deg: Start of the mask azimuth range.
            azimuth_extent_deg: Width of the mask azimuth range.
            return_stats: Whether to attach per-source keep/drop counts.
        """
        if not lidar_masks:
            raise ValueError("lidar_masks must not be empty")
        self.lidar_masks = {normalize_lidar_name(name): mask for name, mask in lidar_masks.items()}
        self.use_calibration_azimuth_offsets = use_calibration_azimuth_offsets
        self.channel_dim = channel_dim if channel_dim is None else int(channel_dim)
        self.azimuth_start_deg = float(azimuth_start_deg)
        self.azimuth_extent_deg = float(azimuth_extent_deg)
        if self.azimuth_extent_deg <= 0.0:
            raise ValueError("azimuth_extent_deg must be positive.")
        self.return_stats = return_stats

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply the per-LiDAR mask and keep aligned per-point arrays consistent."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        keep_mask = np.zeros(points.shape[0], dtype=bool)
        # The vehicle masks the raw sweep, so decide on pre-correction coordinates when the sample
        # carries them. Surviving points keep their original corrected coordinates either way.
        decision_xyz = pre_correction_points(input_dict, points)
        stats = []

        for source in iter_pointcloud_sources(input_dict, points.shape[0]):
            lidar_name = normalize_lidar_name(source.name)
            lidar_mask = self.lidar_masks.get(lidar_name)
            if lidar_mask is None:
                raise KeyError(f"No Nebula mask configured for source {source.name!r}.")
            source_mask = self._source_keep_mask(points, source, lidar_mask, decision_xyz)
            keep_mask[source.point_slice] = source_mask
            if self.return_stats:
                stats.append(
                    {
                        "source_name": source.name,
                        "num_input_points": int(source_mask.size),
                        "num_kept_points": int(source_mask.sum()),
                    }
                )

        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[keep_mask]
        if self.return_stats:
            input_dict["nebula_downsample_stats"] = stats
        return input_dict

    def _source_keep_mask(
        self,
        points: npt.NDArray[np.float32],
        source: SourceSlice,
        lidar_mask: LidarMask,
        decision_xyz: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.bool_]:
        source_rows = points[source.point_slice]
        source_points = (
            source_rows[:, :3] if decision_xyz is None else decision_xyz[source.point_slice]
        )
        local_points = source_points
        if source.translation is not None and source.rotation is not None:
            local_points = (source_points - source.translation) @ source.rotation

        channels = self._channels(source_rows, local_points, lidar_mask)
        azimuth_deg = np.rad2deg(nebula_azimuth_rad(local_points))
        if self.use_calibration_azimuth_offsets:
            azimuth_deg = azimuth_deg - lidar_mask.azimuth_deg[channels]
        azimuth_deg = (azimuth_deg - self.azimuth_start_deg) % 360.0

        keep_grid = lidar_mask.keep
        x = round_half_up(azimuth_deg / self.azimuth_extent_deg * keep_grid.shape[1])
        valid = (
            (x >= 0) & (x < keep_grid.shape[1]) & (channels >= 0) & (channels < keep_grid.shape[0])
        )
        keep = np.zeros(source_points.shape[0], dtype=bool)
        keep[valid] = keep_grid[channels[valid], x[valid]]
        return keep

    def _channels(
        self,
        source_rows: npt.NDArray[np.float32],
        local_points: npt.NDArray[np.float32],
        lidar_mask: LidarMask,
    ) -> npt.NDArray[np.int64]:
        """Prefer the stored ring index; fall back to estimating it from elevation.

        Estimating the ring by nearest calibration elevation is unreliable -- on a recording where
        the true index is available it agrees for only 14-68% of points on four of eight LiDARs,
        because neighbouring Pandar channels are separated by less than the elevation spread of a
        single return. T4Dataset preserves the ring index, so use it whenever it is present.
        """
        if self.channel_dim is not None and source_rows.shape[1] > self.channel_dim:
            return source_rows[:, self.channel_dim].astype(np.int64)
        return nearest_channel(local_points, lidar_mask.elevation_rad)


def nearest_channel(
    points: npt.NDArray[np.float32], elevation_rad: npt.NDArray[np.float32]
) -> npt.NDArray[np.int64]:
    """Estimate each point's ring by nearest calibration elevation."""
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    point_elevation = np.arctan2(points[:, 2], xy_norm)
    return np.abs(point_elevation[:, None] - elevation_rad[None, :]).argmin(axis=1).astype(np.int64)


def nebula_azimuth_rad(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Azimuth in Nebula's convention: measured from +y toward +x."""
    return np.arctan2(points[:, 0], points[:, 1])


def round_half_up(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Round half away from zero, matching the driver rather than numpy's round-half-to-even."""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)
