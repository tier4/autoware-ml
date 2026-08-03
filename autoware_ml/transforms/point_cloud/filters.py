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

"""Point-cloud filters that emulate vehicle-side preprocessing."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


_ASSET_ROOT = resources.files("autoware_ml.configs").joinpath("assets")
_DEFAULT_MASK_ROOT = _ASSET_ROOT.joinpath("aip_x2_gen2")
_DEFAULT_CALIBRATION_ROOT = _ASSET_ROOT.joinpath("hesai")

_DEFAULT_MASKS = {
    name: f"{name}/generated_30deg_roi.param.png"
    for name in (
        "front_upper",
        "front_lower",
        "left_upper",
        "left_lower",
        "rear_upper",
        "rear_lower",
        "right_upper",
        "right_lower",
    )
}
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
_DEFAULT_CALIBRATIONS = {
    "pandar128e4x": "pandar128e4x.csv",
    "pandar_qt128": "pandar_qt128.csv",
}


@dataclass(frozen=True)
class _Calibration:
    elevation_rad: npt.NDArray[np.float32]
    azimuth_deg: npt.NDArray[np.float32]


@dataclass(frozen=True)
class _SourceSlice:
    name: str
    sensor_token: str | None
    point_slice: slice
    translation: npt.NDArray[np.float32] | None
    rotation: npt.NDArray[np.float32] | None


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
        mask_root: str | None = None,
        calibration_root: str | None = None,
        lidar_name_to_mask: Mapping[str, str] | None = None,
        lidar_name_to_model: Mapping[str, str] | None = None,
        model_to_calibration: Mapping[str, str] | None = None,
        use_calibration_azimuth_offsets: bool = True,
        azimuth_start_deg: float = 0.0,
        azimuth_extent_deg: float = 360.0,
        return_stats: bool = False,
    ) -> None:
        """Initialize the Nebula mask-filter approximation.

        Args:
            mask_root: Root directory for mask PNGs. Defaults to bundled J6 Gen2 masks.
            calibration_root: Root directory for calibration CSVs. Defaults to bundled calibration
                files for Pandar128E4X/OT128 and PandarQT128.
            lidar_name_to_mask: LiDAR-name to mask path mapping. Relative paths resolve under
                ``mask_root``.
            lidar_name_to_model: LiDAR-name to model key mapping.
            model_to_calibration: Model key to calibration CSV mapping. Relative paths resolve under
                ``calibration_root``.
            use_calibration_azimuth_offsets: Whether to subtract per-channel azimuth calibration
                offsets before sampling the mask x-coordinate.
            azimuth_start_deg: Start of the mask azimuth range.
            azimuth_extent_deg: Width of the mask azimuth range.
            return_stats: Whether to attach per-source keep/drop counts.
        """
        self.mask_root = Path(mask_root) if mask_root is not None else Path(_DEFAULT_MASK_ROOT)
        self.calibration_root = Path(calibration_root) if calibration_root is not None else Path(
            _DEFAULT_CALIBRATION_ROOT
        )
        self.lidar_name_to_mask = {
            _normalize_lidar_name(name): path
            for name, path in (lidar_name_to_mask or _DEFAULT_MASKS).items()
        }
        self.lidar_name_to_model = {
            _normalize_lidar_name(name): _normalize_model_name(model)
            for name, model in (lidar_name_to_model or _DEFAULT_MODELS).items()
        }
        self.model_to_calibration = {
            _normalize_model_name(model): path
            for model, path in (model_to_calibration or _DEFAULT_CALIBRATIONS).items()
        }
        self.use_calibration_azimuth_offsets = use_calibration_azimuth_offsets
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
        stats = []

        for source in self._iter_sources(input_dict, points.shape[0]):
            lidar_name = _normalize_lidar_name(source.name)
            model_name = self.lidar_name_to_model.get(lidar_name)
            if model_name is None:
                raise KeyError(f"No Nebula lidar model configured for source {source.name!r}.")
            source_mask = self._source_keep_mask(points, source, lidar_name, model_name)
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

    def _iter_sources(self, input_dict: Mapping[str, Any], point_count: int) -> list[_SourceSlice]:
        lidar_sources = input_dict.get("lidar_sources")
        lidar_sources_info = input_dict.get("lidar_sources_info")
        if isinstance(lidar_sources, Mapping) and isinstance(lidar_sources_info, Mapping):
            return self._iter_concat_sources(lidar_sources, lidar_sources_info)

        source_name = input_dict.get("source_name") or input_dict.get("lidar_source_name")
        if source_name is None:
            sample_name = str(input_dict.get("name", ""))
            source_name = _infer_lidar_name_from_text(sample_name)
        if source_name is None:
            raise KeyError(
                "NebulaDownsampleMaskFilter requires concat 'lidar_sources' metadata or a "
                "single-source 'source_name'."
            )
        return [
            _SourceSlice(
                name=str(source_name),
                sensor_token=input_dict.get("sensor_token"),
                point_slice=slice(0, point_count),
                translation=None,
                rotation=None,
            )
        ]

    def _iter_concat_sources(
        self, lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
    ) -> list[_SourceSlice]:
        source_ranges = {
            str(source.get("sensor_token")): source
            for source in lidar_sources_info.get("sources", [])
        }
        output = []
        for source_name, source_meta in lidar_sources.items():
            sensor_token = str(source_meta.get("sensor_token"))
            source_range = source_ranges.get(sensor_token)
            if source_range is None:
                continue
            idx_begin = int(source_range["idx_begin"])
            length = int(source_range["length"])
            output.append(
                _SourceSlice(
                    name=str(source_name),
                    sensor_token=sensor_token,
                    point_slice=slice(idx_begin, idx_begin + length),
                    translation=np.asarray(source_meta.get("translation"), dtype=np.float32),
                    rotation=np.asarray(source_meta.get("rotation"), dtype=np.float32),
                )
            )
        if not output:
            raise ValueError("lidar_sources_info did not match any configured lidar_sources.")
        return output

    def _source_keep_mask(
        self,
        points: npt.NDArray[np.float32],
        source: _SourceSlice,
        lidar_name: str,
        model_name: str,
    ) -> npt.NDArray[np.bool_]:
        source_points = points[source.point_slice, :3]
        local_points = source_points
        if source.translation is not None and source.rotation is not None:
            local_points = (source_points - source.translation) @ source.rotation

        calibration = self._load_calibration(model_name)
        channels = _nearest_channel(local_points, calibration.elevation_rad)
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

    def _load_mask(self, lidar_name: str, model_name: str) -> npt.NDArray[np.bool_]:
        if lidar_name in self._masks:
            return self._masks[lidar_name]
        mask_path = _resolve_path(self.mask_root, self.lidar_name_to_mask[lidar_name])
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read Nebula downsample mask: {mask_path}")
        calibration = self._load_calibration(model_name)
        if image.shape[0] != calibration.elevation_rad.shape[0]:
            raise ValueError(
                f"Mask {mask_path} has {image.shape[0]} rows, but {model_name} calibration has "
                f"{calibration.elevation_rad.shape[0]} channels."
            )
        self._masks[lidar_name] = _dither_mask(image, model_name)
        return self._masks[lidar_name]

    def _load_calibration(self, model_name: str) -> _Calibration:
        if model_name in self._calibrations:
            return self._calibrations[model_name]
        calibration_path = _resolve_path(
            self.calibration_root, self.model_to_calibration[model_name]
        )
        elevations = []
        azimuths = []
        with open(calibration_path, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                elevations.append(float(row["Elevation"]))
                azimuths.append(float(row["Azimuth"]))
        if not elevations:
            raise ValueError(f"Calibration file has no channel rows: {calibration_path}")
        self._calibrations[model_name] = _Calibration(
            elevation_rad=np.deg2rad(np.asarray(elevations, dtype=np.float32)),
            azimuth_deg=np.asarray(azimuths, dtype=np.float32),
        )
        return self._calibrations[model_name]


def _dither_mask(image: npt.NDArray[np.uint8], model_name: str) -> npt.NDArray[np.bool_]:
    height, width = image.shape
    y, x = np.indices((height, width), dtype=np.int64)
    if model_name == "pandar128e4x":
        positions = (
            ((x // 2) * 2 + (y // 4) * 4 + (y % 2))
            % NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS
        )
    else:
        positions = (x + y) % NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS

    numerator = image.astype(np.uint32) * NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS // 255
    output = np.zeros(image.shape, dtype=bool)
    for keep_count in range(1, NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS + 1):
        kept_positions = _round_half_up(
            NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS / float(keep_count)
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


def _normalize_lidar_name(name: str) -> str:
    normalized = str(name).lower()
    if normalized.startswith("lidar_"):
        normalized = normalized[len("lidar_") :]
    return normalized


def _normalize_model_name(name: str) -> str:
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"ot128", "pandar128", "pandar128e4x"}:
        return "pandar128e4x"
    if normalized in {"qt128", "pandarqt128", "pandar_qt128"}:
        return "pandar_qt128"
    return normalized


def _infer_lidar_name_from_text(text: str) -> str | None:
    normalized = text.lower()
    for lidar_name in _DEFAULT_MASKS:
        if lidar_name in normalized:
            return lidar_name
    return None


def _resolve_path(root: Path, path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return root / resolved


def _round_half_up(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)
