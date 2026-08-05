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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

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
        return iter_concat_sources(lidar_sources, lidar_sources_info)

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
    lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
) -> list[SourceSlice]:
    """Pair each configured LiDAR with its index range in the concatenated cloud.

    Raises:
        ValueError: No configured source matched the recorded ranges.
    """
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
            SourceSlice(
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
