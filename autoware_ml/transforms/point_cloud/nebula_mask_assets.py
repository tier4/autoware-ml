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
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from autoware_ml.transforms.point_cloud.assets import ASSET_ROOT, resolve_asset_path
from autoware_ml.transforms.point_cloud.lidar_sources import normalize_lidar_name
from autoware_ml.transforms.point_cloud.nebula_mask import LidarMask, round_half_up

# Nebula quantises mask intensity into this many levels and dithers between them.
QUANTIZATION_LEVELS = 10

DEFAULT_CALIBRATION_ROOT = ASSET_ROOT / "hesai"
# Which LiDAR carries which sensor, and which mask belongs to it, is platform knowledge. It lives in
# this file beside each platform's masks rather than as a default here, so that pointing the loader
# at one platform's directory cannot quietly apply another's sensor models and dither patterns.
MANIFEST_NAME = "lidar_masks.param.yaml"


def load_lidar_masks(
    mask_root: str | Path,
    *,
    calibration_root: str | Path | None = None,
    lidars: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, LidarMask]:
    """Read a platform's per-LiDAR masks and pair each with its calibration.

    Args:
        mask_root: Directory holding the per-LiDAR masks and their manifest. A relative path
            resolves under the bundled asset root, so the name of a bundled platform directory is
            enough.
        calibration_root: Directory holding the calibration CSVs. Defaults to the bundled Hesai
            tables.
        lidars: LiDAR name to ``{mask, model, calibration}``, overriding the manifest in
            ``mask_root``. ``calibration`` is optional and defaults to the model's own table, so a
            LiDAR cannot end up holding one model's dither pattern and another's elevations.

    Returns:
        Masks keyed by mounting position, ready to pass to
        :class:`~autoware_ml.transforms.point_cloud.nebula_mask.NebulaDownsampleMaskFilter`.
    """
    mask_dir = resolve_asset_path(mask_root)
    entries = read_mask_manifest(mask_dir) if lidars is None else lidars
    calibration_dir = (
        DEFAULT_CALIBRATION_ROOT if calibration_root is None else Path(calibration_root)
    )

    loaded: dict[str, LidarMask] = {}
    calibration_cache: dict[str, npt.NDArray[np.float32]] = {}
    for raw_name, entry in entries.items():
        lidar_name = normalize_lidar_name(raw_name)
        try:
            mask_path = entry["mask"]
            model_name = normalize_model_name(entry["model"])
        except KeyError as error:
            raise KeyError(
                f"LiDAR {lidar_name!r} needs both 'mask' and 'model'; got {sorted(entry)}."
            ) from error
        calibration_file = entry.get("calibration") or f"{model_name}.csv"

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
    if not loaded:
        raise ValueError(f"No LiDARs configured for mask root {mask_dir}.")
    return loaded


def read_mask_manifest(mask_dir: Path) -> dict[str, dict[str, Any]]:
    """Read a platform's ``lidar_masks.param.yaml``.

    Raises:
        FileNotFoundError: The directory carries no manifest, so the platform's sensor layout is
            unknown and there is no safe default to fall back on.
    """
    manifest = Path(mask_dir) / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {mask_dir}. It records which sensor model and mask each LiDAR "
            "of the platform carries; pass 'lidars' explicitly if the platform has no manifest."
        )
    document = yaml.safe_load(manifest.read_text()) or {}
    entries = document.get("lidars")
    if not entries:
        raise ValueError(f"No 'lidars' entry in {manifest}")
    return {str(name): dict(entry) for name, entry in entries.items()}


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

    Raises:
        ValueError: The model is not one this reproduces a pattern for. Falling back to a pattern
            that happens to be implemented would return a plausible mask that is simply wrong,
            which no later check would catch.
    """
    height, width = image.shape
    y, x = np.indices((height, width), dtype=np.int64)
    match normalize_model_name(model_name):
        case "pandar128e4x":
            positions = ((x // 2) * 2 + (y // 4) * 4 + (y % 2)) % QUANTIZATION_LEVELS
        case "pandar_qt128":
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
    """Map the aliases used for Hesai models onto the two calibration tables that exist.

    Raises:
        ValueError: The name is not a known alias of either model. A typo has to fail here: every
            later step -- the dither pattern, the elevation table, the channel-count check that both
            128-row tables pass -- would accept the wrong sensor and produce a mask nothing flags.
    """
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"ot128", "pandar128", "pandar128e4x"}:
        return "pandar128e4x"
    if normalized in {"qt128", "pandarqt128", "pandar_qt128"}:
        return "pandar_qt128"
    raise ValueError(
        f"Unsupported Hesai model {name!r}. Reproduced models are pandar128e4x (aliases ot128, "
        "pandar128) and pandar_qt128 (aliases qt128, pandarqt128)."
    )
