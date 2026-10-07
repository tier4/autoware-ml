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

"""Reader for the t4pack LiDAR frame container.

A scene may keep the frames of a LiDAR channel, ``data/<channel>/*.pcd.bin``, as one
``data/<channel>.pack`` file instead. Each frame is compressed on its own with zstd over its
columns, the float columns byte-shuffled, and a JSON index at the end of the file names every
frame with its byte range, point count and column types (format ``t4pack`` v1).

Packs are written by the data processing tools, not by autoware-ml. Record generation reads
the index of a scene's pack once and stores the location of each frame in the record table,
so loading a frame reads one byte range and does not parse the index again.
"""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np
import zstandard
from jaxtyping import Float32
from pydantic import BaseModel, ConfigDict

T4PACK_MAGIC = b"T4PACK\x00\x01"
T4PACK_SUFFIX = ".pack"
# Index offset and index size, followed by the magic again, end the file
_TRAILER = struct.Struct("<QQ")
# One open file per pack and process; dataloader workers are processes
_MAX_OPEN_PACKS = 32


class T4PackFrameLocation(BaseModel):
    """
    Location of one frame inside its pack.

    Attributes:
      offset: Byte offset of the compressed frame in the pack.
      size: Byte size of the compressed frame.
      num_points: Number of points of the frame.
      dtypes: t4pack type code of every column: ``f4s`` byte-shuffled float32, ``u1`` uint8,
        ``i1`` int8.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    offset: int
    size: int
    num_points: int
    dtypes: tuple[str, ...]


def t4pack_path(point_cloud_path: str) -> str:
    """
    Return the pack that holds a frame, ``data/<channel>.pack`` for ``data/<channel>/<name>``.

    Args:
      point_cloud_path: Path of the frame's ``.pcd.bin`` file, which need not exist.

    Returns:
      str: Path of the channel pack.
    """

    return os.path.dirname(point_cloud_path) + T4PACK_SUFFIX


def read_t4pack_index(pack_path: str) -> Mapping[str, T4PackFrameLocation]:
    """
    Read the frame index of a pack.

    Args:
      pack_path: Path of the pack.

    Returns:
      Mapping[str, T4PackFrameLocation]: Location of every frame, keyed by its file name.

    Raises:
      ValueError: Raised when the file is not a t4pack v1 file.
    """

    with open(pack_path, "rb") as reader:
        reader.seek(0, os.SEEK_END)
        size = reader.tell()
        trailer_size = _TRAILER.size + len(T4PACK_MAGIC)
        reader.seek(max(size - trailer_size, 0))
        trailer = reader.read(trailer_size)
        reader.seek(0)
        if (
            len(trailer) != trailer_size
            or trailer[_TRAILER.size :] != T4PACK_MAGIC
            or reader.read(len(T4PACK_MAGIC)) != T4PACK_MAGIC
        ):
            raise ValueError(f"{pack_path} is not a t4pack file.")
        index_offset, index_size = _TRAILER.unpack(trailer[: _TRAILER.size])
        reader.seek(index_offset)
        index = json.loads(zstandard.ZstdDecompressor().decompress(reader.read(index_size)))

    if index.get("format") != "t4pack" or index.get("version") != 1:
        raise ValueError(
            f"{pack_path} is a {index.get('format')} v{index.get('version')} file, "
            "only t4pack v1 is supported."
        )
    return {
        frame["name"]: T4PackFrameLocation(
            offset=int(frame["offset"]),
            size=int(frame["size"]),
            num_points=int(frame["n_points"]),
            dtypes=tuple(frame["dtypes"]),
        )
        for frame in index["frames"]
    }


def decode_t4pack_frame(
    raw: bytes, dtypes: Sequence[str], num_points: int
) -> Float32[np.ndarray, "num_points num_columns"]:
    """
    Decode the uncompressed columns of one frame.

    Args:
      raw: Decompressed frame, its columns one after the other.
      dtypes: t4pack type code of every column.
      num_points: Number of points of the frame.

    Returns:
      Float32[np.ndarray, "num_points num_columns"]: The points, as ``np.fromfile`` returns
        the loose ``.pcd.bin`` file.
    """

    points = np.empty((num_points, len(dtypes)), dtype=np.float32)
    position = 0
    for column, dtype in enumerate(dtypes):
        if dtype == "f4s":
            # The four bytes of every float are stored as four planes
            planes = np.frombuffer(raw, np.uint8, 4 * num_points, position).reshape(4, num_points)
            points[:, column] = np.ascontiguousarray(planes.T).reshape(-1).view(np.float32)
            position += 4 * num_points
        elif dtype == "u1":
            points[:, column] = np.frombuffer(raw, np.uint8, num_points, position)
            position += num_points
        elif dtype == "i1":
            points[:, column] = np.frombuffer(raw, np.int8, num_points, position)
            position += num_points
        else:
            raise ValueError(f"Column {column} has the unknown t4pack type {dtype!r}.")
    if position != len(raw):
        raise ValueError(f"The frame has {len(raw) - position} bytes after its columns.")
    return points


_open_packs: OrderedDict[str, int] = OrderedDict()


def _pack_file_descriptor(pack_path: str) -> int:
    """Return the process-local descriptor of a pack, closing the least recently used one."""
    descriptor = _open_packs.get(pack_path)
    if descriptor is None:
        descriptor = os.open(pack_path, os.O_RDONLY)
        _open_packs[pack_path] = descriptor
        while len(_open_packs) > _MAX_OPEN_PACKS:
            _, stale = _open_packs.popitem(last=False)
            os.close(stale)
    else:
        _open_packs.move_to_end(pack_path)
    return descriptor


def read_t4pack_frame(
    pack_path: str, location: T4PackFrameLocation, num_features: int
) -> Float32[np.ndarray, "num_points num_features"]:
    """
    Read one frame of a pack.

    Args:
      pack_path: Path of the pack.
      location: Location of the frame, as the record table stores it.
      num_features: Number of features the record declares per point.

    Returns:
      Float32[np.ndarray, "num_points num_features"]: The points of the frame.

    Raises:
      ValueError: Raised when the frame does not have ``num_features`` columns or is truncated.
    """

    if len(location.dtypes) != num_features:
        raise ValueError(
            f"The frame at offset {location.offset} of {pack_path} has {len(location.dtypes)} "
            f"columns, the record declares {num_features} features."
        )
    blob = os.pread(_pack_file_descriptor(pack_path), location.size, location.offset)
    if len(blob) != location.size:
        raise ValueError(
            f"{pack_path} ends before the frame at offset {location.offset} of {location.size} "
            "bytes."
        )
    raw = zstandard.ZstdDecompressor().decompress(blob)
    return decode_t4pack_frame(raw, location.dtypes, location.num_points)


__all__ = [
    "T4PackFrameLocation",
    "decode_t4pack_frame",
    "read_t4pack_frame",
    "read_t4pack_index",
    "t4pack_path",
]
