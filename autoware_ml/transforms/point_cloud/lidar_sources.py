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

"""Split a concatenated pointcloud back into the LiDARs it came from.

T4Dataset stores one concatenated cloud per frame, with ``lidar_sources_info`` recording which
index range each contributing LiDAR occupies and ``lidar_sources`` carrying its extrinsics. Filters
that reproduce per-LiDAR vehicle-side behaviour need those ranges to work sensor by sensor.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# T4 writes one of these beside every concatenated cloud, recording how that cloud was assembled.
CONCAT_INFO_DIRECTORY = "LIDAR_CONCAT_INFO"

# Mounting positions used to name per-LiDAR assets and to recognise a source from free text.
LIDAR_POSITION_NAMES = (
    "front_upper",
    "front_lower",
    "left_upper",
    "left_lower",
    "rear_upper",
    "rear_lower",
    "right_upper",
    "right_lower",
)


@dataclass(frozen=True)
class SourceSlice:
    """One LiDAR's contribution to a concatenated cloud."""

    name: str
    sensor_token: str | None
    point_slice: slice
    translation: npt.NDArray[np.float32] | None
    rotation: npt.NDArray[np.float32] | None


def normalize_lidar_name(name: str) -> str:
    """Reduce a sensor channel name such as ``LIDAR_FRONT_UPPER`` to ``front_upper``."""
    return str(name).lower().removeprefix("lidar_")


def infer_lidar_name_from_text(text: str) -> str | None:
    """Recognise a mounting position mentioned anywhere in ``text``."""
    normalized = text.lower()
    for lidar_name in LIDAR_POSITION_NAMES:
        if lidar_name in normalized:
            return lidar_name
    return None


def sources_info_path(lidar_path: str | Path) -> Path | None:
    """Locate the ``LIDAR_CONCAT_INFO`` sidecar belonging to a concatenated cloud.

    T4 stores one JSON per concatenated cloud, under a sibling directory and sharing its frame
    number. Returns ``None`` when there is none.
    """
    cloud = Path(lidar_path)
    candidate = cloud.parent.parent / CONCAT_INFO_DIRECTORY / f"{cloud.name.split('.')[0]}.json"
    return candidate if candidate.is_file() else None


def resolve_sources_info(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Find the concat metadata for one scan, wherever it happens to be recorded.

    Keyframes carry ``lidar_sources_info`` inline. Sweeps generally do not: infos generated before
    that was propagated name no source metadata at all, so it is read from the sidecar beside the
    cloud, which is byte-identical to what the keyframes carry.

    Returns ``None`` when the scan has no metadata anywhere, leaving the decision to the caller.
    """
    info = entry.get("lidar_sources_info")
    if isinstance(info, Mapping):
        return dict(info)
    explicit = entry.get("lidar_pointcloud_source_path")
    if explicit:
        return json.loads(Path(explicit).read_text())
    lidar_path = entry.get("lidar_path")
    if lidar_path:
        sidecar = sources_info_path(lidar_path)
        if sidecar is not None:
            return json.loads(sidecar.read_text())
    return None


def iter_pointcloud_sources(input_dict: Mapping[str, Any], point_count: int) -> list[SourceSlice]:
    """Return the per-LiDAR slices of a sample's ``points``.

    Concatenated samples are split using their source metadata. A single-source sample yields one
    slice covering every point, named from ``source_name`` or inferred from the sample name.

    Raises:
        KeyError: The sample carries neither concat metadata nor a usable source name.
    """
    lidar_sources = input_dict.get("lidar_sources")
    lidar_sources_info = input_dict.get("lidar_sources_info")
    if isinstance(lidar_sources, Mapping) and isinstance(lidar_sources_info, Mapping):
        return iter_concat_sources(lidar_sources, lidar_sources_info, point_count)

    source_name = input_dict.get("source_name") or input_dict.get("lidar_source_name")
    if source_name is None:
        source_name = infer_lidar_name_from_text(str(input_dict.get("name", "")))
    if source_name is None:
        raise KeyError(
            "Pointcloud source-aware filters require concat 'lidar_sources' metadata or a "
            "single-source 'source_name'."
        )
    return [
        SourceSlice(
            name=str(source_name),
            sensor_token=input_dict.get("sensor_token"),
            point_slice=slice(0, point_count),
            translation=None,
            rotation=None,
        )
    ]


def iter_concat_sources(
    lidar_sources: Mapping[str, Any],
    lidar_sources_info: Mapping[str, Any],
    point_count: int,
) -> list[SourceSlice]:
    """Pair each configured LiDAR with its index range in the concatenated cloud.

    Every point must fall in exactly one configured source's range. Callers decide per source, so a
    point left outside all of them has no decision to be made about it -- a filter would silently
    drop it, or worse judge it against the wrong LiDAR's calibration. That is checked here rather
    than left to each caller.

    A cloud longer than the recorded ranges is the shape of one specific mistake: appending
    historical sweeps, as ``LoadPointsFromMultiSweeps`` does, while ``lidar_sources_info`` goes on
    describing the keyframe alone. Masking a multi-sweep cloud needs per-sweep ranges, so this
    raises instead.

    Raises:
        ValueError: No configured source matched, a range falls outside the cloud, two ranges
            overlap, or the ranges leave points uncovered.
        KeyError: A matched source is missing its extrinsics.
    """
    source_ranges = {
        str(source.get("sensor_token")): source for source in lidar_sources_info.get("sources", [])
    }
    output = []
    for source_name, source_meta in lidar_sources.items():
        sensor_token = str(source_meta.get("sensor_token"))
        source_range = source_ranges.get(sensor_token)
        if source_range is None:
            # A configured LiDAR that contributed nothing to this frame. Its absence is only a
            # problem if it leaves points uncovered, which the coverage check below catches.
            continue
        idx_begin = int(source_range["idx_begin"])
        length = int(source_range["length"])
        if length < 0 or idx_begin < 0 or idx_begin + length > point_count:
            raise ValueError(
                f"lidar_sources_info range {idx_begin}:{idx_begin + length} for source "
                f"{source_name!r} does not fit a cloud of {point_count} points."
            )
        translation = source_meta.get("translation")
        rotation = source_meta.get("rotation")
        if translation is None or rotation is None:
            raise KeyError(
                f"Source {source_name!r} is missing its extrinsics; both 'translation' and "
                "'rotation' are needed to bring its points into the sensor frame."
            )
        output.append(
            SourceSlice(
                name=str(source_name),
                sensor_token=sensor_token,
                point_slice=slice(idx_begin, idx_begin + length),
                translation=np.asarray(translation, dtype=np.float32),
                rotation=np.asarray(rotation, dtype=np.float32),
            )
        )
    if not output:
        raise ValueError("lidar_sources_info did not match any configured lidar_sources.")
    _assert_partitions(output, point_count)
    return output


def _assert_partitions(sources: list[SourceSlice], point_count: int) -> None:
    """Check the slices tile ``0:point_count`` exactly once each."""
    covered = 0
    previous_end = 0
    for source in sorted(sources, key=lambda item: item.point_slice.start):
        start, stop = source.point_slice.start, source.point_slice.stop
        if start < previous_end:
            raise ValueError(
                f"lidar_sources_info ranges overlap at index {start}; source {source.name!r} "
                f"starts inside a range that runs to {previous_end}."
            )
        previous_end = stop
        covered += stop - start
    if covered != point_count:
        raise ValueError(
            f"lidar_sources_info covers {covered} of {point_count} points. Every point must belong "
            "to exactly one source. If the cloud carries appended sweeps, the source ranges "
            "describe only the keyframe and per-sweep ranges are needed."
        )
