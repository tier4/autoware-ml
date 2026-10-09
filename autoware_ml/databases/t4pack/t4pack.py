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

import functools
import json
import os
import struct
from collections.abc import Mapping, Sequence
from typing import BinaryIO

import numpy as np
import zstandard
from jaxtyping import Float32

from autoware_ml.databases.t4pack.t4pack_frame import T4PackFrame


class T4Pack:
    """
    One t4pack file, holding every frame of one LiDAR channel of a scene.

    Attributes:
      pack_path: Path of the pack.
    """

    MAGIC = b"T4PACK\x00\x01"
    SUFFIX = ".pack"
    # Index offset and index size, followed by the magic again, end the file
    TRAILER = struct.Struct("<QQ")
    # Open files kept per process; dataloader workers are processes
    MAX_OPEN_PACKS = 32

    def __init__(self, pack_path: str) -> None:
        """
        Initialize the reader. The file is opened when it is first read.

        Args:
          pack_path: Path of the pack.
        """

        self.pack_path = pack_path

    @classmethod
    def from_point_cloud_path(cls, point_cloud_path: str) -> T4Pack:
        """
        Return the pack that holds a frame, ``data/<channel>.pack`` for ``data/<channel>/<name>``.

        Args:
          point_cloud_path: Path of the frame's ``.pcd.bin`` file, which need not exist.

        Returns:
          T4Pack: The pack of the frame's channel, which need not exist.
        """

        return cls(os.path.dirname(point_cloud_path) + cls.SUFFIX)

    def read_index(self) -> Mapping[str, T4PackFrame]:
        """
        Read the frame index of the pack.

        Returns:
          Mapping[str, T4PackFrame]: Location of every frame, keyed by its file name.

        Raises:
          ValueError: Raised when the file is not a t4pack v1 file.
        """

        with open(self.pack_path, "rb") as reader:
            reader.seek(0, os.SEEK_END)
            size = reader.tell()
            trailer_size = self.TRAILER.size + len(self.MAGIC)
            reader.seek(max(size - trailer_size, 0))
            trailer = reader.read(trailer_size)
            reader.seek(0)
            if (
                len(trailer) != trailer_size
                or trailer[self.TRAILER.size :] != self.MAGIC
                or reader.read(len(self.MAGIC)) != self.MAGIC
            ):
                raise ValueError(f"{self.pack_path} is not a t4pack file.")
            index_offset, index_size = self.TRAILER.unpack(trailer[: self.TRAILER.size])
            reader.seek(index_offset)
            index = json.loads(zstandard.ZstdDecompressor().decompress(reader.read(index_size)))

        if index.get("format") != "t4pack" or index.get("version") != 1:
            raise ValueError(
                f"{self.pack_path} is a {index.get('format')} v{index.get('version')} file, "
                "only t4pack v1 is supported."
            )
        return {
            frame["name"]: T4PackFrame(
                offset=int(frame["offset"]),
                size=int(frame["size"]),
                num_points=int(frame["n_points"]),
                dtypes=tuple(frame["dtypes"]),
            )
            for frame in index["frames"]
        }

    def read_frame(
        self, frame: T4PackFrame, num_features: int
    ) -> Float32[np.ndarray, "num_points num_features"]:
        """
        Read one frame of the pack.

        Args:
          frame: Location of the frame, as the record table stores it.
          num_features: Number of features the record declares per point.

        Returns:
          Float32[np.ndarray, "num_points num_features"]: The points of the frame.

        Raises:
          ValueError: Raised when the frame does not have ``num_features`` columns or is truncated.
        """

        if len(frame.dtypes) != num_features:
            raise ValueError(
                f"The frame at offset {frame.offset} of {self.pack_path} has "
                f"{len(frame.dtypes)} columns, the record declares {num_features} features."
            )
        blob = os.pread(self._file_descriptor(), frame.size, frame.offset)
        if len(blob) != frame.size:
            raise ValueError(
                f"{self.pack_path} ends before the frame at offset {frame.offset} of "
                f"{frame.size} bytes."
            )
        raw = zstandard.ZstdDecompressor().decompress(blob)
        return self.decode_frame(raw, frame.dtypes, frame.num_points)

    @staticmethod
    def decode_frame(
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
                planes = np.frombuffer(raw, np.uint8, 4 * num_points, position)
                planes = planes.reshape(4, num_points)
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

    def _file_descriptor(self) -> int:
        """Return the process-local descriptor of the pack, opening it on first use."""
        return self._open_file(self.pack_path).fileno()

    @staticmethod
    @functools.lru_cache(maxsize=MAX_OPEN_PACKS)
    def _open_file(pack_path: str) -> BinaryIO:
        """Open a pack once per process; the least recently used file is closed when evicted."""
        return open(pack_path, "rb")
