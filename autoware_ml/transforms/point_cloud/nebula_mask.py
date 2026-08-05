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

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.assets import ASSET_ROOT, resolve_asset_path
from autoware_ml.transforms.point_cloud.ego_motion import pre_correction_points
from autoware_ml.transforms.point_cloud.lidar_sources import (
    LIDAR_POSITION_NAMES,
    SourceSlice,
    iter_concat_sources,
    iter_pointcloud_sources,
    normalize_lidar_name,
)

_DEFAULT_CALIBRATION_ROOT = ASSET_ROOT / "hesai"
_DEFAULT_MASKS = {name: f"{name}/generated_30deg_roi.param.png" for name in LIDAR_POSITION_NAMES}
# Sensor model per mounting position. Upper mounts carry a long-range Pandar, lower mounts a
# short-range QT; the model selects which calibration table the mask is indexed against.
_DEFAULT_MODELS = {
    "front_upper": "pandar128e4x",
    "left_upper": "pandar128e4x",
    "rear_upper": "pandar128e4x",
    "right_upper": "pandar128e4x",
    "front_lower": "pandar_qt128",
    "left_lower": "pandar_qt128",
    "rear_lower": "pandar_qt128",
    "right_lower": "pandar_qt128",
}
_DEFAULT_CALIBRATIONS = {name: f"{model}.csv" for name, model in _DEFAULT_MODELS.items()}


@dataclass(frozen=True)
class _Calibration:
    elevation_rad: npt.NDArray[np.float32]
    azimuth_deg: npt.NDArray[np.float32]


class NebulaDownsampleMaskFilter(BaseTransform):
    """Approximate Nebula's decode-time downsample mask on loaded Cartesian points.

    The vehicle filter runs in packet/range-view space before ego-motion correction. This transform
    reconstructs a per-LiDAR azimuth/channel index from Cartesian points, applies the same bundled
    dithered mask, and keeps the original point rows in their existing coordinate frame.

    Required keys:
        points: Raw point array of shape ``(N, D)``.

    Optional keys:
        lidar_sources: Mapping from LiDAR name to calibration/extrinsic metadata.
        lidar_sources_info: PointCloudMetainfo-like dictionary containing source slices.
        source_name: Name of the single source when the sample has already been sliced.
        translation, rotation: Single-source extrinsics used only for diagnostics; points are
            assumed to already be in that source frame when no concat metadata is present.

    Generated keys:
        points and all aligned per-point numpy arrays filtered to the kept rows.
        nebula_downsample_stats when ``return_stats`` is true.
    """

    _required_keys = ["points"]
    _QUANTIZATION_LEVELS = 10

    def __init__(
        self,
        *,
        mask_root: str,
        calibration_root: str | None = None,
        lidar_name_to_mask: Mapping[str, str] | None = None,
        lidar_name_to_model: Mapping[str, str] | None = None,
        lidar_name_to_calibration: Mapping[str, str] | None = None,
        use_calibration_azimuth_offsets: bool = False,
        channel_dim: int | None = 4,
        azimuth_start_deg: float = 0.0,
        azimuth_extent_deg: float = 360.0,
        return_stats: bool = False,
    ) -> None:
        """Initialize the Nebula mask-filter approximation.

        Args:
            mask_root: Directory holding the per-LiDAR mask PNGs. A relative path resolves under
                the bundled asset root, so the name of a bundled platform directory is enough.
            calibration_root: Root directory for calibration CSVs. Defaults to bundled calibration
                files for Pandar128E4X/OT128 and PandarQT128.
            lidar_name_to_mask: LiDAR-name to mask path mapping. Relative paths resolve under
                ``mask_root``.
            lidar_name_to_model: LiDAR-name to model key mapping.
            lidar_name_to_calibration: LiDAR-name to calibration CSV mapping. Relative paths resolve
                under ``calibration_root``.
            use_calibration_azimuth_offsets: Whether to subtract per-channel azimuth calibration
                offsets before sampling the mask x-coordinate. Defaults to ``False``: the driver
                indexes the mask by raw azimuth, and subtracting the offsets moves points into the
                wrong mask column.
            channel_dim: Index of the per-point feature holding the ring/channel number. Set to
                ``None`` to always estimate the ring from elevation instead.
            azimuth_start_deg: Start of the mask azimuth range.
            azimuth_extent_deg: Width of the mask azimuth range.
            return_stats: Whether to attach per-source keep/drop counts.
        """
        self.mask_root = resolve_asset_path(mask_root)
        self.calibration_root = (
            Path(calibration_root)
            if calibration_root is not None
            else Path(_DEFAULT_CALIBRATION_ROOT)
        )
        self.lidar_name_to_mask = {
            normalize_lidar_name(name): path
            for name, path in (lidar_name_to_mask or _DEFAULT_MASKS).items()
        }
        self.lidar_name_to_model = {
            normalize_lidar_name(name): _normalize_model_name(model)
            for name, model in (lidar_name_to_model or _DEFAULT_MODELS).items()
        }
        self.lidar_name_to_calibration = {
            normalize_lidar_name(name): path
            for name, path in (lidar_name_to_calibration or _DEFAULT_CALIBRATIONS).items()
        }
        self.use_calibration_azimuth_offsets = use_calibration_azimuth_offsets
        self.channel_dim = channel_dim if channel_dim is None else int(channel_dim)
        self.azimuth_start_deg = float(azimuth_start_deg)
        self.azimuth_extent_deg = float(azimuth_extent_deg)
        if self.azimuth_extent_deg <= 0.0:
            raise ValueError("azimuth_extent_deg must be positive.")
        self.return_stats = return_stats
        self._masks: dict[str, npt.NDArray[np.bool_]] = {}
        self._calibrations: dict[str, _Calibration] = {}

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply the per-LiDAR mask and keep aligned per-point arrays consistent."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        keep_mask = np.zeros(points.shape[0], dtype=bool)
        # The vehicle masks the raw sweep, so decide on pre-correction coordinates when the sample
        # carries them. Surviving points keep their original corrected coordinates either way.
        decision_xyz = pre_correction_points(input_dict, points)
        stats = []

        for source in self._iter_sources(input_dict, points.shape[0]):
            lidar_name = normalize_lidar_name(source.name)
            model_name = self.lidar_name_to_model.get(lidar_name)
            if model_name is None:
                raise KeyError(f"No Nebula lidar model configured for source {source.name!r}.")
            source_mask = self._source_keep_mask(
                points, source, lidar_name, model_name, decision_xyz
            )
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

    def _iter_sources(self, input_dict: Mapping[str, Any], point_count: int) -> list[SourceSlice]:
        return iter_pointcloud_sources(input_dict, point_count)

    def iter_concat_sources(
        self, lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
    ) -> list[SourceSlice]:
        return iter_concat_sources(lidar_sources, lidar_sources_info)

    def _source_keep_mask(
        self,
        points: npt.NDArray[np.float32],
        source: SourceSlice,
        lidar_name: str,
        model_name: str,
        decision_xyz: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.bool_]:
        source_rows = points[source.point_slice]
        source_points = (
            source_rows[:, :3] if decision_xyz is None else decision_xyz[source.point_slice]
        )
        local_points = source_points
        if source.translation is not None and source.rotation is not None:
            local_points = (source_points - source.translation) @ source.rotation

        calibration = self._load_calibration(lidar_name)
        channels = self._channels(source_rows, local_points, calibration)
        azimuth_deg = np.rad2deg(_nebula_azimuth_rad(local_points))
        if self.use_calibration_azimuth_offsets:
            azimuth_deg = azimuth_deg - calibration.azimuth_deg[channels]
        azimuth_deg = (azimuth_deg - self.azimuth_start_deg) % 360.0

        mask = self._load_mask(lidar_name, model_name)
        x = _round_half_up(azimuth_deg / self.azimuth_extent_deg * mask.shape[1])
        valid = (x >= 0) & (x < mask.shape[1]) & (channels >= 0) & (channels < mask.shape[0])
        keep = np.zeros(source_points.shape[0], dtype=bool)
        keep[valid] = mask[channels[valid], x[valid]]
        return keep

    def _channels(
        self,
        source_rows: npt.NDArray[np.float32],
        local_points: npt.NDArray[np.float32],
        calibration: _Calibration,
    ) -> npt.NDArray[np.int64]:
        """Prefer the stored ring index; fall back to estimating it from elevation.

        Estimating the ring by nearest calibration elevation is unreliable -- on a recording where
        the true index is available it agrees for only 14-68% of points on four of eight LiDARs,
        because neighbouring Pandar channels are separated by less than the elevation spread of a
        single return. T4Dataset preserves the ring index, so use it whenever it is present.
        """
        if self.channel_dim is not None and source_rows.shape[1] > self.channel_dim:
            return source_rows[:, self.channel_dim].astype(np.int64)
        return _nearest_channel(local_points, calibration.elevation_rad)

    def _load_mask(self, lidar_name: str, model_name: str) -> npt.NDArray[np.bool_]:
        if lidar_name in self._masks:
            return self._masks[lidar_name]
        mask_path = resolve_asset_path(self.lidar_name_to_mask[lidar_name], self.mask_root)
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read Nebula downsample mask: {mask_path}")
        calibration = self._load_calibration(lidar_name)
        if image.shape[0] != calibration.elevation_rad.shape[0]:
            raise ValueError(
                f"Mask {mask_path} has {image.shape[0]} rows, but {model_name} calibration has "
                f"{calibration.elevation_rad.shape[0]} channels."
            )
        self._masks[lidar_name] = _dither_mask(image, model_name)
        return self._masks[lidar_name]

    def _load_calibration(self, lidar_name: str) -> _Calibration:
        lidar_name = normalize_lidar_name(lidar_name)
        if lidar_name in self._calibrations:
            return self._calibrations[lidar_name]
        calibration_path = resolve_asset_path(
            self.lidar_name_to_calibration[lidar_name], self.calibration_root
        )
        elevations = []
        azimuths = []
        with open(calibration_path, newline="") as file:
            rows = csv.reader(file)
            header = None
            for row in rows:
                if "Elevation" in row and "Azimuth" in row:
                    header = row
                    break
            if header is None:
                raise ValueError(
                    f"Calibration file has no Elevation/Azimuth header: {calibration_path}"
                )
            reader = csv.DictReader(file, fieldnames=header)
            for row in reader:
                if not row.get("Elevation") or not row.get("Azimuth"):
                    continue
                elevations.append(float(row["Elevation"]))
                azimuths.append(float(row["Azimuth"]))
        if not elevations:
            raise ValueError(f"Calibration file has no channel rows: {calibration_path}")
        self._calibrations[lidar_name] = _Calibration(
            elevation_rad=np.deg2rad(np.asarray(elevations, dtype=np.float32)),
            azimuth_deg=np.asarray(azimuths, dtype=np.float32),
        )
        return self._calibrations[lidar_name]


def _dither_mask(image: npt.NDArray[np.uint8], model_name: str) -> npt.NDArray[np.bool_]:
    height, width = image.shape
    y, x = np.indices((height, width), dtype=np.int64)
    if model_name == "pandar128e4x":
        positions = (
            (x // 2) * 2 + (y // 4) * 4 + (y % 2)
        ) % NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS
    else:
        positions = (x + y) % NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS

    numerator = image.astype(np.uint32) * NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS // 255
    output = np.zeros(image.shape, dtype=bool)
    for keep_count in range(1, NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS + 1):
        kept_positions = _round_half_up(
            NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS
            / float(keep_count)
            * np.arange(keep_count)
        )
        output |= (numerator == keep_count) & np.isin(positions, kept_positions)
    return output


def _nearest_channel(
    points: npt.NDArray[np.float32], elevation_rad: npt.NDArray[np.float32]
) -> npt.NDArray[np.int64]:
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    point_elevation = np.arctan2(points[:, 2], xy_norm)
    return np.abs(point_elevation[:, None] - elevation_rad[None, :]).argmin(axis=1).astype(np.int64)


def _nebula_azimuth_rad(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.arctan2(points[:, 0], points[:, 1])


def _normalize_model_name(name: str) -> str:
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"ot128", "pandar128", "pandar128e4x"}:
        return "pandar128e4x"
    if normalized in {"qt128", "pandarqt128", "pandar_qt128"}:
        return "pandar_qt128"
    return normalized


def _round_half_up(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)
