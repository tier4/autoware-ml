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

"""Test-only writer of small t4pack files.

autoware-ml only reads packs; the data processing tools write them. This writer follows the
same layout so the reader can be tested without those tools.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import zstandard

from autoware_ml.databases.t4pack.t4pack import T4Pack

#: Column types of a T4 LiDAR frame: x, y, z byte-shuffled float32, intensity u8, ring i8.
LIDAR_DTYPES = ("f4s", "f4s", "f4s", "u1", "i1")


def encode_columns(frame: np.ndarray, dtypes: Sequence[str]) -> bytes:
    """Lay out the columns of one ``[N, len(dtypes)]`` float32 frame one after the other."""
    parts = []
    for column, dtype in enumerate(dtypes):
        values = frame[:, column]
        if dtype == "f4s":
            planes = np.ascontiguousarray(values, np.float32).view(np.uint8).reshape(-1, 4)
            parts.append(np.ascontiguousarray(planes.T).tobytes())
        elif dtype == "u1":
            parts.append(values.astype(np.uint8).tobytes())
        else:
            parts.append(values.astype(np.int8).tobytes())
    return b"".join(parts)


def write_test_pack(
    path: Path, frames: Mapping[str, np.ndarray], dtypes: Sequence[str] = LIDAR_DTYPES
) -> None:
    """Write named frames as one t4pack v1 file, in the given order."""
    compressor = zstandard.ZstdCompressor(level=1)
    entries = []
    with path.open("wb") as writer:
        writer.write(T4Pack.MAGIC)
        for name, frame in frames.items():
            blob = compressor.compress(encode_columns(np.asarray(frame, np.float32), dtypes))
            entries.append(
                {
                    "name": name,
                    "offset": writer.tell(),
                    "size": len(blob),
                    "n_points": int(len(frame)),
                    "dtypes": list(dtypes),
                }
            )
            writer.write(blob)
        index = compressor.compress(
            json.dumps(
                {"format": "t4pack", "version": 1, "n_frames": len(entries), "frames": entries}
            ).encode()
        )
        index_offset = writer.tell()
        writer.write(index)
        writer.write(T4Pack.TRAILER.pack(index_offset, len(index)))
        writer.write(T4Pack.MAGIC)


def random_lidar_frame(rng: np.random.Generator, num_points: int) -> np.ndarray:
    """Return a frame whose intensity and ring fit their u8 and i8 columns."""
    frame = np.empty((num_points, 5), dtype=np.float32)
    frame[:, :3] = rng.normal(0.0, 50.0, (num_points, 3)).astype(np.float32)
    frame[:, 3] = rng.integers(0, 256, num_points)
    frame[:, 4] = rng.integers(-1, 8, num_points)
    return frame
