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

"""Reader for the t4pack LiDAR sweep container.

Storage-constrained hosts keep a scene's ``data/LIDAR_CONCAT/*.pcd.bin`` frames as one
``data/LIDAR_CONCAT.pack`` file instead: every frame zstd-compressed on its own over
byte-shuffled float columns, plus a JSON index (format ``t4pack`` v1, the container of
TIER IV's e2e-devkit, re-implemented here so the loader does not depend on it).

The packing tool wrote the frames in the sorted order of the loose ``*.pcd.bin`` names
and named frame ``i`` ``{i:05d}.pcd.bin`` in the index. T4 scenes name their frames by
a gapless five digit sequence, so those index names are the real file names, and a
frame is looked up by name. When a name is not in the index (a corpus with other file
names) the frame's rank among the sorted LiDAR file names of the scene's
``annotation/sample_data.json`` is used instead, which is exact only when that table
lists every packed frame; the count is checked. :func:`read_packed_frame` caches the
per-scene name tables and open readers per process, which is what one dataloader
worker is.
"""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float32

MAGIC = b"T4PACK\x00\x01"
TRAILER = struct.Struct("<QQ")
PACK_SUFFIX = ".pack"

#: Column dtypes of a T4 LiDAR sweep: x, y, z byte-shuffled f32, intensity u8, ring i8.
LIDAR_COLUMNS = ("f4s", "f4s", "f4s", "u1", "i1")
LIDAR_FIELDS = ("x", "y", "z", "intensity", "ring")

_MAX_OPEN_READERS = 32


def _zstd():  # type: ignore[no-untyped-def]
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading packed LiDAR frames needs the 'zstandard' package; install the project "
            "dependencies."
        ) from error
    return zstandard


def encode_frame(frame: np.ndarray, dtypes: Sequence[str] = LIDAR_COLUMNS) -> bytes:
    """Serialize one ``[N, len(dtypes)]`` float32 frame to an uncompressed column blob.

    Args:
        frame: Point matrix.
        dtypes: t4pack dtype code per column.

    Returns:
        The concatenated column planes.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[1] != len(dtypes):
        raise ValueError(f"frame must be [N, {len(dtypes)}], got {frame.shape}")
    parts = []
    for column, dtype in enumerate(dtypes):
        values = frame[:, column]
        if dtype == "f4s":
            planes = np.ascontiguousarray(values, np.float32).view(np.uint8).reshape(-1, 4)
            parts.append(np.ascontiguousarray(planes.T).tobytes())
        elif dtype == "u1":
            parts.append(values.astype(np.uint8).tobytes())
        elif dtype == "i1":
            parts.append(values.astype(np.int8).tobytes())
        else:
            raise ValueError(f"column {column}: unknown t4pack dtype {dtype!r}")
    return b"".join(parts)


def decode_frame(raw: bytes, dtypes: Sequence[str], n_points: int) -> np.ndarray:
    """Inverse of :func:`encode_frame`.

    Args:
        raw: Uncompressed column blob.
        dtypes: t4pack dtype code per column.
        n_points: Number of points of the frame.

    Returns:
        ``[n_points, len(dtypes)]`` float32 point matrix.
    """
    n = int(n_points)
    out = np.empty((n, len(dtypes)), dtype=np.float32)
    pos = 0
    for column, dtype in enumerate(dtypes):
        if dtype == "f4s":
            planes = np.frombuffer(raw, np.uint8, 4 * n, pos).reshape(4, n)
            out[:, column] = np.ascontiguousarray(planes.T).reshape(-1).view(np.float32)
            pos += 4 * n
        elif dtype == "u1":
            out[:, column] = np.frombuffer(raw, np.uint8, n, pos)
            pos += n
        elif dtype == "i1":
            out[:, column] = np.frombuffer(raw, np.int8, n, pos)
            pos += n
        else:
            raise ValueError(f"column {column}: unknown t4pack dtype {dtype!r}")
    if pos != len(raw):
        raise ValueError(f"frame has {len(raw) - pos} trailing bytes")
    return out


def write_pack(
    path: str | Path,
    frames: Sequence[np.ndarray],
    dtypes: Sequence[str] = LIDAR_COLUMNS,
    fields: Sequence[str] = LIDAR_FIELDS,
    level: int = 1,
) -> int:
    """Write frames as one t4pack v1 file, atomically.

    Args:
        path: Destination file.
        frames: ``[N_i, len(dtypes)]`` float32 frames in the order they are indexed.
        dtypes: t4pack dtype code per column.
        fields: Field names recorded in the index.
        level: zstd compression level.

    Returns:
        Number of frames written.
    """
    if len(fields) != len(dtypes):
        raise ValueError(f"{len(fields)} field names for {len(dtypes)} columns")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    compressor = _zstd().ZstdCompressor(level=level)
    entries = []
    try:
        with tmp.open("wb") as writer:
            writer.write(MAGIC)
            for index, frame in enumerate(frames):
                frame = np.asarray(frame, dtype=np.float32)
                blob = compressor.compress(encode_frame(frame, dtypes))
                entries.append(
                    {
                        "name": f"{index:05d}.pcd.bin",
                        "offset": writer.tell(),
                        "size": len(blob),
                        "n_points": int(frame.shape[0]),
                        "dtypes": list(dtypes),
                    }
                )
                writer.write(blob)
            index_blob = compressor.compress(
                json.dumps(
                    {
                        "format": "t4pack",
                        "version": 1,
                        "n_frames": len(entries),
                        "fields": list(fields),
                        "frames": entries,
                    },
                    separators=(",", ":"),
                ).encode()
            )
            index_offset = writer.tell()
            writer.write(index_blob)
            writer.write(TRAILER.pack(index_offset, len(index_blob)))
            writer.write(MAGIC)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return len(entries)


class PackReader:
    """Random access reader for one ``.pack`` file.

    Not thread safe: one file descriptor and one decompressor per instance. Dataloader
    workers are processes, so one reader per process per pack is the intended use.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self._dctx = _zstd().ZstdDecompressor()
        try:
            size = self.path.stat().st_size
            trailer_size = TRAILER.size + len(MAGIC)
            trailer = os.pread(self._fd, trailer_size, max(size - trailer_size, 0))
            if (
                len(trailer) != trailer_size
                or trailer[TRAILER.size :] != MAGIC
                or os.pread(self._fd, len(MAGIC), 0) != MAGIC
            ):
                raise ValueError(f"{self.path}: not a t4pack file")
            index_offset, index_size = TRAILER.unpack(trailer[: TRAILER.size])
            index = json.loads(self._dctx.decompress(os.pread(self._fd, index_size, index_offset)))
            if index.get("format") != "t4pack" or index.get("version") != 1:
                raise ValueError(
                    f"{self.path}: unsupported pack {index.get('format')}/{index.get('version')}"
                )
            self.frames = index["frames"]
            self.fields = tuple(index.get("fields") or ())
            self.n_frames = int(index["n_frames"])
            if self.n_frames != len(self.frames):
                raise ValueError(f"{self.path}: n_frames does not match the frame index")
            self.index_by_name = {
                str(frame.get("name")): position for position, frame in enumerate(self.frames)
            }
        except Exception:
            self.close()
            raise

    def read_frame(self, index: int) -> np.ndarray:
        """Decode frame ``index`` as ``[n_points, n_columns]`` float32."""
        index = int(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"{self.path}: frame {index} of {self.n_frames}")
        frame = self.frames[index]
        blob = os.pread(self._fd, int(frame["size"]), int(frame["offset"]))
        return decode_frame(
            self._dctx.decompress(blob), tuple(frame["dtypes"]), int(frame["n_points"])
        )

    def __len__(self) -> int:
        return self.n_frames

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the file descriptor."""
        fd = getattr(self, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            self._fd = -1

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()


@lru_cache(maxsize=256)
def scene_frame_names(scene_dir: str, channel_dir: str) -> tuple[str, ...]:
    """Return the sorted LiDAR file names of one channel of a T4 scene.

    The names come from the scene's ``annotation/sample_data.json``, the table every
    loose frame of a T4 scene is listed in, so they reproduce the order the packing tool
    saw in the loose directory.

    Args:
        scene_dir: Version directory of the scene, the parent of ``data/``.
        channel_dir: Channel directory below ``data/``, e.g. ``LIDAR_CONCAT``.

    Returns:
        Sorted base names of the channel's ``*.pcd.bin`` files.
    """
    table = Path(scene_dir) / "annotation" / "sample_data.json"
    if not table.is_file():
        raise FileNotFoundError(
            f"Cannot index the packed frames of {scene_dir}: no annotation/sample_data.json."
        )
    prefix = f"data/{channel_dir}/"
    names = {
        os.path.basename(entry["filename"])
        for entry in json.loads(table.read_text())
        if str(entry.get("filename", "")).startswith(prefix)
        and str(entry["filename"]).endswith(".pcd.bin")
    }
    return tuple(sorted(names))


_readers: OrderedDict[str, PackReader] = OrderedDict()


def pack_reader(pack_path: str) -> PackReader:
    """Return the process-local reader of a pack, opening it on first use."""
    reader = _readers.get(pack_path)
    if reader is None:
        reader = PackReader(pack_path)
        _readers[pack_path] = reader
        while len(_readers) > _MAX_OPEN_READERS:
            _, stale = _readers.popitem(last=False)
            stale.close()
    else:
        _readers.move_to_end(pack_path)
    return reader


def pack_path_for(loose_path: str) -> str:
    """Return the pack that replaces the directory of a loose frame path."""
    return os.path.dirname(loose_path) + PACK_SUFFIX


def packed_frame_index(loose_path: str, num_features: int | None = None) -> tuple[PackReader, int]:
    """Resolve a loose frame path to its pack reader and frame index.

    Args:
        loose_path: Absolute path the record names, ``<scene>/data/<channel>/<name>.pcd.bin``.
        num_features: Number of columns the record declares, checked against the pack when
            given.

    Returns:
        The process-local reader of the pack and the index of the frame in it.

    Raises:
        FileNotFoundError: Raised when the pack does not exist or does not hold the frame.
        ValueError: Raised when the pack does not match the scene's frame table or the
            requested feature count.
    """
    pack_path = pack_path_for(loose_path)
    if not os.path.isfile(pack_path):
        raise FileNotFoundError(
            f"Neither the point cloud {loose_path} nor its pack {pack_path} exists."
        )
    frame_path = Path(loose_path)
    scene_dir, channel_dir = str(frame_path.parents[2]), frame_path.parent.name
    reader = pack_reader(pack_path)
    if num_features is not None and len(reader.fields) != num_features:
        raise ValueError(
            f"{pack_path} stores {len(reader.fields)} columns {reader.fields}, the record expects "
            f"{num_features} features."
        )
    names = scene_frame_names(scene_dir, channel_dir)
    if frame_path.name not in names:
        raise FileNotFoundError(f"{frame_path.name} is not a {channel_dir} frame of {scene_dir}.")
    # The index names are the real file names when every name the scene table lists is
    # among them (T4 frames are a gapless five digit sequence, the index names as well).
    if all(name in reader.index_by_name for name in names):
        return reader, reader.index_by_name[frame_path.name]
    if reader.n_frames != len(names):
        raise ValueError(
            f"{pack_path} holds {reader.n_frames} frames whose index names do not cover the "
            f"{len(names)} {channel_dir} frames the sample_data table of {scene_dir} lists, so "
            "the frames cannot be matched by name or by rank."
        )
    return reader, names.index(frame_path.name)


def read_packed_frame(
    loose_path: str, num_features: int
) -> Float32[np.ndarray, "num_points num_features"]:
    """Read a frame whose loose ``.pcd.bin`` was replaced by the channel pack.

    Args:
        loose_path: Absolute path the record names, ``<scene>/data/<channel>/<name>.pcd.bin``.
        num_features: Number of columns the record declares for the frame.

    Returns:
        The point matrix, as ``np.fromfile(...).reshape(-1, num_features)`` would return it.
    """
    reader, index = packed_frame_index(loose_path, num_features)
    return reader.read_frame(index)


def load_point_cloud_file(
    path: str, num_features: int
) -> Float32[np.ndarray, "num_points num_features"]:
    """Read a ``.pcd.bin`` frame, from its channel pack when the loose file is absent.

    Args:
        path: Path of the loose frame file the record names.
        num_features: Number of float32 columns per point.

    Returns:
        The point matrix of shape ``(num_points, num_features)``.
    """
    if os.path.isfile(path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, num_features)
    # Storage constrained hosts replace a scene's loose frames by one t4pack file
    return read_packed_frame(path, num_features)
