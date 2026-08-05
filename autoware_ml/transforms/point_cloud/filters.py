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
_DEFAULT_CALIBRATIONS = {name: f"{_DEFAULT_MODELS[name]}.csv" for name in _DEFAULT_MASKS}


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
        lidar_name_to_calibration: Mapping[str, str] | None = None,
        use_calibration_azimuth_offsets: bool = False,
        channel_dim: int | None = 4,
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
        self.mask_root = Path(mask_root) if mask_root is not None else Path(_DEFAULT_MASK_ROOT)
        self.calibration_root = (
            Path(calibration_root)
            if calibration_root is not None
            else Path(_DEFAULT_CALIBRATION_ROOT)
        )
        self.lidar_name_to_mask = {
            _normalize_lidar_name(name): path
            for name, path in (lidar_name_to_mask or _DEFAULT_MASKS).items()
        }
        self.lidar_name_to_model = {
            _normalize_lidar_name(name): _normalize_model_name(model)
            for name, model in (lidar_name_to_model or _DEFAULT_MODELS).items()
        }
        self.lidar_name_to_calibration = {
            _normalize_lidar_name(name): path
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
        return _iter_pointcloud_sources(input_dict, point_count)

    def _iter_concat_sources(
        self, lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
    ) -> list[_SourceSlice]:
        return _iter_concat_sources(lidar_sources, lidar_sources_info)

    def _source_keep_mask(
        self,
        points: npt.NDArray[np.float32],
        source: _SourceSlice,
        lidar_name: str,
        model_name: str,
    ) -> npt.NDArray[np.bool_]:
        source_rows = points[source.point_slice]
        source_points = source_rows[:, :3]
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
        mask_path = _resolve_path(self.mask_root, self.lidar_name_to_mask[lidar_name])
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
        lidar_name = _normalize_lidar_name(lidar_name)
        if lidar_name in self._calibrations:
            return self._calibrations[lidar_name]
        calibration_path = _resolve_path(
            self.calibration_root, self.lidar_name_to_calibration[lidar_name]
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


class EgoCropBoxFilter(BaseTransform):
    """Remove points falling inside the ego vehicle's crop boxes.

    Autoware crops the vehicle body and the steered front wheels out of every LiDAR scan before
    concatenation (two negative crop boxes, see ``nebula_node_container.launch.py``). T4Dataset
    carries un-cropped concatenated clouds -- the ego vehicle is plainly visible in them -- so this
    step has to be reapplied to match the inference-time point distribution. It removes up to 18%
    of a single LiDAR's points (``rear_lower`` on AIP X2 Gen2).

    Ordering matters. On the vehicle the crop is evaluated as a *mask* that is only AND-ed into the
    output at the very end, so cropped points are still present as neighbours while the ring outlier
    filter runs. Place this transform **after** :class:`RingOutlierFilter`; running it first
    discards those neighbours and measurably changes the ring filter's decisions.

    Points are expected in the ego/``base_link`` frame, which is how T4Dataset stores them.
    """

    _required_keys = ["points"]

    def __init__(self, *, crop_boxes: Sequence[Sequence[float]] | None = None) -> None:
        """Initialize the EgoCropBoxFilter transform.

        Args:
            crop_boxes: Boxes to remove, each ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in the
                ego frame. Defaults to the AIP X2 Gen2 self and wheels boxes.
        """
        boxes = AIP_X2_GEN2_EGO_CROP_BOXES if crop_boxes is None else crop_boxes
        self.crop_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
        if self.crop_boxes.size == 0:
            raise ValueError("crop_boxes must contain at least one box")

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Drop points inside any configured box, keeping aligned per-point arrays consistent."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        inside = np.zeros(points.shape[0], dtype=bool)
        for x_min, y_min, z_min, x_max, y_max, z_max in self.crop_boxes:
            inside |= (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )

        keep_mask = ~inside
        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[keep_mask]
        return input_dict


def ego_crop_boxes_from_vehicle_info(
    *,
    wheel_base: float,
    wheel_tread: float,
    wheel_radius: float,
    wheel_width: float,
    front_overhang: float,
    rear_overhang: float,
    left_overhang: float,
    right_overhang: float,
    vehicle_height: float,
    max_steer_angle: float,
) -> list[list[float]]:
    """Derive the self and wheels crop boxes from vehicle dimensions.

    Transcribes ``get_vehicle_info()`` in
    ``aip_x2_gen2_launch/launch/nebula_node_container.launch.py``, so the boxes stay consistent with
    whatever the vehicle's ``vehicle_info.param.yaml`` declares.

    Args:
        wheel_base: Distance between front and rear wheel centres.
        wheel_tread: Distance between left and right wheel centres.
        wheel_radius: Wheel radius.
        wheel_width: Wheel width.
        front_overhang: Front wheel centre to vehicle front.
        rear_overhang: Rear wheel centre to vehicle rear.
        left_overhang: Left wheel centre to vehicle left.
        right_overhang: Right wheel centre to vehicle right.
        vehicle_height: Overall vehicle height.
        max_steer_angle: Maximum tire cut angle in radians.

    Returns:
        Two boxes as ``[x_min, y_min, z_min, x_max, y_max, z_max]``: the vehicle body, then the
        swept volume of the steered front wheels.
    """
    half_width = wheel_width / 2.0
    center_to_corner = float(np.hypot(half_width, wheel_radius))
    corner_angle = float(np.arctan2(half_width, wheel_radius))
    if corner_angle < max_steer_angle:
        max_longitudinal = center_to_corner
    else:
        max_longitudinal = center_to_corner * float(np.cos(max_steer_angle - corner_angle))
    max_lateral = center_to_corner * float(np.sin(max_steer_angle + corner_angle))

    self_box = [
        -rear_overhang,
        -(wheel_tread / 2.0 + right_overhang),
        0.0,
        front_overhang + wheel_base,
        wheel_tread / 2.0 + left_overhang,
        vehicle_height,
    ]
    # The wheel box is scaled to 110% of wheel diameter upstream to absorb suspension travel.
    wheels_box = [
        wheel_base - max_longitudinal,
        -(wheel_tread / 2.0 + max_lateral),
        0.0,
        wheel_base + max_longitudinal,
        wheel_tread / 2.0 + max_lateral,
        wheel_radius * 2.2,
    ]
    return [self_box, wheels_box]


# Derived from j6_gen2_description/config/vehicle_info.param.yaml.
AIP_X2_GEN2_EGO_CROP_BOXES = ego_crop_boxes_from_vehicle_info(
    wheel_base=4.76012,
    wheel_tread=1.754,
    wheel_radius=0.3725,
    wheel_width=0.215,
    front_overhang=0.95099,
    rear_overhang=1.52579,
    left_overhang=0.32358,
    right_overhang=0.34983,
    vehicle_height=3.080,
    max_steer_angle=0.838,
)


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


def _iter_pointcloud_sources(input_dict: Mapping[str, Any], point_count: int) -> list[_SourceSlice]:
    lidar_sources = input_dict.get("lidar_sources")
    lidar_sources_info = input_dict.get("lidar_sources_info")
    if isinstance(lidar_sources, Mapping) and isinstance(lidar_sources_info, Mapping):
        return _iter_concat_sources(lidar_sources, lidar_sources_info)

    source_name = input_dict.get("source_name") or input_dict.get("lidar_source_name")
    if source_name is None:
        sample_name = str(input_dict.get("name", ""))
        source_name = _infer_lidar_name_from_text(sample_name)
    if source_name is None:
        raise KeyError(
            "Pointcloud source-aware filters require concat 'lidar_sources' metadata or a "
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
    lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
) -> list[_SourceSlice]:
    source_ranges = {
        str(source.get("sensor_token")): source for source in lidar_sources_info.get("sources", [])
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
