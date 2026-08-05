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

"""Load Nebula downsample masks and Hesai calibration from disk.

Kept apart from the transform so that
:class:`~autoware_ml.transforms.point_cloud.nebula_mask.NebulaDownsampleMaskFilter` performs no file
access and can be constructed from in-memory masks.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.point_cloud.assets import ASSET_ROOT, resolve_asset_path
from autoware_ml.transforms.point_cloud.lidar_sources import (
    LIDAR_POSITION_NAMES,
    normalize_lidar_name,
)
from autoware_ml.transforms.point_cloud.nebula_mask import LidarMask, round_half_up

# Nebula quantises mask intensity into this many levels and dithers between them.
QUANTIZATION_LEVELS = 10

DEFAULT_CALIBRATION_ROOT = ASSET_ROOT / "hesai"
DEFAULT_MASKS = {name: f"{name}/generated_30deg_roi.param.png" for name in LIDAR_POSITION_NAMES}
# Sensor model per mounting position. Upper mounts carry a long-range Pandar, lower mounts a
# short-range QT; the model selects both the calibration table and the dither pattern.
DEFAULT_MODELS = {
    "front_upper": "pandar128e4x",
    "left_upper": "pandar128e4x",
    "rear_upper": "pandar128e4x",
    "right_upper": "pandar128e4x",
    "front_lower": "pandar_qt128",
    "left_lower": "pandar_qt128",
    "rear_lower": "pandar_qt128",
    "right_lower": "pandar_qt128",
}
DEFAULT_CALIBRATIONS = {name: f"{model}.csv" for name, model in DEFAULT_MODELS.items()}


def load_lidar_masks(
    mask_root: str | Path,
    *,
    calibration_root: str | Path | None = None,
    lidar_name_to_mask: Mapping[str, str] | None = None,
    lidar_name_to_model: Mapping[str, str] | None = None,
    lidar_name_to_calibration: Mapping[str, str] | None = None,
) -> dict[str, LidarMask]:
    """Read a platform's per-LiDAR masks and pair each with its calibration.

    Args:
        mask_root: Directory holding the per-LiDAR mask PNGs. A relative path resolves under the
            bundled asset root, so the name of a bundled platform directory is enough.
        calibration_root: Directory holding the calibration CSVs. Defaults to the bundled Hesai
            tables.
        lidar_name_to_mask: LiDAR name to mask path, relative to ``mask_root``.
        lidar_name_to_model: LiDAR name to sensor model, which selects the dither pattern.
        lidar_name_to_calibration: LiDAR name to calibration CSV, relative to ``calibration_root``.

    Returns:
        Masks keyed by mounting position, ready to pass to
        :class:`~autoware_ml.transforms.point_cloud.nebula_mask.NebulaDownsampleMaskFilter`.
    """
    masks = _normalize_keys(lidar_name_to_mask or DEFAULT_MASKS)
    models = {
        name: normalize_model_name(model)
        for name, model in _normalize_keys(lidar_name_to_model or DEFAULT_MODELS).items()
    }
    calibrations = _normalize_keys(lidar_name_to_calibration or DEFAULT_CALIBRATIONS)

    mask_dir = resolve_asset_path(mask_root)
    calibration_dir = (
        DEFAULT_CALIBRATION_ROOT if calibration_root is None else Path(calibration_root)
    )

    loaded: dict[str, LidarMask] = {}
    calibration_cache: dict[str, npt.NDArray[np.float32]] = {}
    for lidar_name, mask_path in masks.items():
        model_name = models.get(lidar_name)
        if model_name is None:
            raise KeyError(f"No sensor model configured for LiDAR {lidar_name!r}.")
        calibration_file = calibrations.get(lidar_name)
        if calibration_file is None:
            raise KeyError(f"No calibration configured for LiDAR {lidar_name!r}.")

        if calibration_file not in calibration_cache:
            calibration_cache[calibration_file] = load_calibration(
                resolve_asset_path(calibration_file, calibration_dir)
            )
        elevation_rad = calibration_cache[calibration_file]

        resolved_mask = resolve_asset_path(mask_path, mask_dir)
        image = cv2.imread(str(resolved_mask), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read Nebula downsample mask: {resolved_mask}")
        if image.shape[0] != elevation_rad.shape[0]:
            raise ValueError(
                f"Mask {resolved_mask} has {image.shape[0]} rows, but {model_name} calibration has "
                f"{elevation_rad.shape[0]} channels."
            )
        loaded[lidar_name] = LidarMask(
            keep=dither_mask(image, model_name),
            elevation_rad=elevation_rad,
        )
    return loaded


def load_calibration(path: str | Path) -> npt.NDArray[np.float32]:
    """Read per-channel elevation from a Hesai calibration CSV.

    Tolerates the metadata rows Hesai ships ahead of the column header. The file also carries a
    per-channel azimuth correction, which is deliberately ignored: the driver does not apply it when
    building the downsample mask.

    Returns:
        Elevation in radians, one entry per channel.
    """
    elevations: list[float] = []
    with open(path, newline="") as file:
        header = None
        for row in csv.reader(file):
            if "Elevation" in row:
                header = row
                break
        if header is None:
            raise ValueError(f"Calibration file has no Elevation header: {path}")
        for row in csv.DictReader(file, fieldnames=header):
            if not row.get("Elevation"):
                continue
            elevations.append(float(row["Elevation"]))
    if not elevations:
        raise ValueError(f"Calibration file has no channel rows: {path}")
    return np.deg2rad(np.asarray(elevations, dtype=np.float32))


def dither_mask(image: npt.NDArray[np.uint8], model_name: str) -> npt.NDArray[np.bool_]:
    """Expand a greyscale keep-ratio image into the driver's per-cell keep/drop decision.

    Pixel intensity encodes what fraction of returns to keep. The driver spreads that fraction over
    a repeating pattern of cells so the survivors are distributed rather than clustered; the pattern
    differs per sensor model.
    """
    height, width = image.shape
    y, x = np.indices((height, width), dtype=np.int64)
    if model_name == "pandar128e4x":
        positions = ((x // 2) * 2 + (y // 4) * 4 + (y % 2)) % QUANTIZATION_LEVELS
    else:
        positions = (x + y) % QUANTIZATION_LEVELS

    numerator = image.astype(np.uint32) * QUANTIZATION_LEVELS // 255
    output = np.zeros(image.shape, dtype=bool)
    for keep_count in range(1, QUANTIZATION_LEVELS + 1):
        kept_positions = round_half_up(
            QUANTIZATION_LEVELS / float(keep_count) * np.arange(keep_count)
        )
        output |= (numerator == keep_count) & np.isin(positions, kept_positions)
    return output


def normalize_model_name(name: str) -> str:
    """Map the aliases used for Hesai models onto the two calibration tables that exist."""
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"ot128", "pandar128", "pandar128e4x"}:
        return "pandar128e4x"
    if normalized in {"qt128", "pandarqt128", "pandar_qt128"}:
        return "pandar_qt128"
    return normalized


def _normalize_keys(mapping: Mapping[str, str]) -> dict[str, str]:
    return {normalize_lidar_name(name): value for name, value in mapping.items()}
