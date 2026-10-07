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

"""Tests for the t4pack reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoware_ml.utils.point_cloud.t4pack import (
    T4PackFrameLocation,
    read_t4pack_frame,
    read_t4pack_index,
    t4pack_path,
)
from autoware_ml.utils.tests.t4pack_fixtures import random_lidar_frame, write_test_pack


def test_frames_read_back_bit_exact(tmp_path: Path) -> None:
    """Every frame decodes to the float32 matrix that was packed, bit for bit."""
    rng = np.random.default_rng(0)
    frames = {f"{i:05d}.pcd.bin": random_lidar_frame(rng, 100 + i) for i in range(3)}
    pack = tmp_path / "LIDAR_CONCAT.pack"
    write_test_pack(pack, frames)

    index = read_t4pack_index(str(pack))

    assert list(index) == list(frames)
    for name, frame in frames.items():
        points = read_t4pack_frame(str(pack), index[name], 5)
        assert points.dtype == np.float32
        assert np.array_equal(points.view(np.uint32), frame.view(np.uint32))


def test_float_intensity_column_keeps_values_above_255(tmp_path: Path) -> None:
    """A pack that stores intensity as float returns intensities a u8 column cannot hold."""
    rng = np.random.default_rng(1)
    frame = random_lidar_frame(rng, 64)
    frame[:, 3] = rng.uniform(0.0, 65535.0, 64).astype(np.float32)
    pack = tmp_path / "LIDAR_CONCAT.pack"
    write_test_pack(pack, {"00000.pcd.bin": frame}, dtypes=("f4s", "f4s", "f4s", "f4s", "i1"))

    location = read_t4pack_index(str(pack))["00000.pcd.bin"]

    assert np.array_equal(read_t4pack_frame(str(pack), location, 5), frame)


def test_index_keeps_the_file_names_of_the_scene(tmp_path: Path) -> None:
    """Frames named without zero padding are indexed under their own names."""
    rng = np.random.default_rng(2)
    names = ["0.pcd.bin", "1.pcd.bin", "10.pcd.bin", "-1-01.pcd.bin"]
    frames = {name: random_lidar_frame(rng, 10 + i) for i, name in enumerate(names)}
    pack = tmp_path / "LIDAR_CONCAT.pack"
    write_test_pack(pack, frames)

    index = read_t4pack_index(str(pack))

    assert sorted(index) == sorted(names)
    assert np.array_equal(
        read_t4pack_frame(str(pack), index["10.pcd.bin"], 5), frames["10.pcd.bin"]
    )


def test_a_file_that_is_not_a_pack_is_rejected(tmp_path: Path) -> None:
    """The index reader checks the magic at both ends of the file."""
    path = tmp_path / "LIDAR_CONCAT.pack"
    path.write_bytes(b"\x00" * 64)

    with pytest.raises(ValueError, match="not a t4pack file"):
        read_t4pack_index(str(path))


def test_a_frame_with_another_column_count_is_rejected(tmp_path: Path) -> None:
    """The record's feature count must match the columns of the packed frame."""
    pack = tmp_path / "LIDAR_CONCAT.pack"
    write_test_pack(pack, {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(3), 8)})
    location = read_t4pack_index(str(pack))["00000.pcd.bin"]

    with pytest.raises(ValueError, match="the record declares 4 features"):
        read_t4pack_frame(str(pack), location, 4)


def test_a_location_past_the_end_of_the_pack_is_rejected(tmp_path: Path) -> None:
    """A stale location, e.g. from a pack that was rewritten since, does not read garbage."""
    pack = tmp_path / "LIDAR_CONCAT.pack"
    write_test_pack(pack, {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(4), 8)})
    location = T4PackFrameLocation(
        offset=pack.stat().st_size - 4, size=64, num_points=8, dtypes=("f4s",) * 3 + ("u1", "i1")
    )

    with pytest.raises(ValueError, match="ends before the frame"):
        read_t4pack_frame(str(pack), location, 5)


def test_the_pack_of_a_frame_replaces_its_channel_directory() -> None:
    """``data/<channel>/<name>`` is held by ``data/<channel>.pack``."""
    assert (
        t4pack_path("/db/scene/0/data/LIDAR_CONCAT/00012.pcd.bin")
        == "/db/scene/0/data/LIDAR_CONCAT.pack"
    )
