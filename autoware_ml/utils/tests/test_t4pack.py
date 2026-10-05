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

"""Tests for the t4pack reader and the packed frame fallback of the point loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoware_ml.utils.point_cloud import t4pack

pytest.importorskip("zstandard")


def _frame(seed: int, num_points: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.empty((num_points, 5), dtype=np.float32)
    frame[:, :3] = rng.uniform(-120, 120, (num_points, 3)).astype(np.float32)
    frame[:, 3] = rng.integers(0, 256, num_points)  # intensity, stored as uint8
    frame[:, 4] = rng.integers(0, 128, num_points)  # ring, stored as int8
    return frame


def _scene(
    tmp_path: Path,
    names: list[str],
    frames: list[np.ndarray],
    unlisted: list[np.ndarray] | None = None,
) -> Path:
    """Write a scene with a sample_data table and the packed channel, no loose frames.

    ``unlisted`` frames are packed after the listed ones without a table entry, like the
    surplus sweeps at the end of a T4 scene.
    """
    scene = tmp_path / "db" / "scene" / "0"
    (scene / "annotation").mkdir(parents=True)
    (scene / "data").mkdir()
    entries = [{"filename": f"data/LIDAR_CONCAT/{name}"} for name in names]
    entries.append({"filename": "data/CAM_FRONT/000.jpg"})
    (scene / "annotation" / "sample_data.json").write_text(json.dumps(entries))
    # The packing tool wrote the frames in sorted file name order.
    order = sorted(range(len(names)), key=lambda i: names[i])
    packed = [frames[i] for i in order] + list(unlisted or [])
    t4pack.write_pack(scene / "data" / "LIDAR_CONCAT.pack", packed)
    return scene


def test_pack_round_trips_frames_bit_exact(tmp_path: Path) -> None:
    frames = [_frame(0, 1000), _frame(1, 0), _frame(2, 37)]
    written = t4pack.write_pack(tmp_path / "x.pack", frames, level=3)
    assert written == 3
    with t4pack.PackReader(tmp_path / "x.pack") as reader:
        assert len(reader) == 3
        assert reader.fields == t4pack.LIDAR_FIELDS
        for index, frame in enumerate(frames):
            assert np.array_equal(reader.read_frame(index), frame)
        with pytest.raises(IndexError):
            reader.read_frame(3)
    with pytest.raises(ValueError, match="not a t4pack file"):
        (tmp_path / "bad.pack").write_bytes(b"nope")
        t4pack.PackReader(tmp_path / "bad.pack")


def test_sequence_named_frames_are_found_by_the_index_name(tmp_path: Path) -> None:
    # T4 frames: a gapless five digit sequence; the scene table lists the first three of
    # five packed frames, the two surplus sweeps at the end have no table entry.
    names = ["00000.pcd.bin", "00001.pcd.bin", "00002.pcd.bin"]
    frames = [_frame(20, 40), _frame(21, 41), _frame(22, 42)]
    scene = _scene(tmp_path, names, frames, unlisted=[_frame(23, 43), _frame(24, 44)])
    t4pack.scene_frame_names.cache_clear()
    t4pack._readers.clear()
    for name, frame in zip(names, frames):
        loose = str(scene / "data" / "LIDAR_CONCAT" / name)
        assert np.array_equal(t4pack.read_packed_frame(loose, 5), frame)
    with pytest.raises(FileNotFoundError, match="not a LIDAR_CONCAT frame"):
        t4pack.read_packed_frame(str(scene / "data" / "LIDAR_CONCAT" / "00004.pcd.bin"), 5)


def test_packed_frames_are_found_by_their_rank_in_the_sample_data_table(tmp_path: Path) -> None:
    names = ["1700000002.pcd.bin", "1700000000.pcd.bin", "1700000001.pcd.bin"]
    frames = [_frame(10, 50), _frame(11, 60), _frame(12, 70)]
    scene = _scene(tmp_path, names, frames)
    t4pack.scene_frame_names.cache_clear()
    t4pack._readers.clear()
    for name, frame in zip(names, frames):
        loose = str(scene / "data" / "LIDAR_CONCAT" / name)
        assert np.array_equal(t4pack.read_packed_frame(loose, 5), frame)
    with pytest.raises(FileNotFoundError, match="not a LIDAR_CONCAT frame"):
        t4pack.read_packed_frame(str(scene / "data" / "LIDAR_CONCAT" / "999.pcd.bin"), 5)
    with pytest.raises(ValueError, match="expects 7 features"):
        t4pack.read_packed_frame(str(scene / "data" / "LIDAR_CONCAT" / names[0]), 7)


def test_pack_that_does_not_match_the_scene_table_is_rejected(tmp_path: Path) -> None:
    scene = _scene(tmp_path, ["a.pcd.bin", "b.pcd.bin"], [_frame(0, 5), _frame(1, 6)])
    # Overwrite the pack with a different frame count.
    t4pack.write_pack(scene / "data" / "LIDAR_CONCAT.pack", [_frame(0, 5)])
    t4pack.scene_frame_names.cache_clear()
    t4pack._readers.clear()
    with pytest.raises(ValueError, match="cannot be matched by name or by rank"):
        t4pack.read_packed_frame(str(scene / "data" / "LIDAR_CONCAT" / "a.pcd.bin"), 5)


def test_frame_loader_prefers_the_loose_file_and_falls_back_to_the_pack(tmp_path: Path) -> None:
    names = ["0.pcd.bin", "1.pcd.bin"]
    frames = [_frame(3, 20), _frame(4, 30)]
    scene = _scene(tmp_path, names, frames)
    t4pack.scene_frame_names.cache_clear()
    t4pack._readers.clear()
    loose_dir = scene / "data" / "LIDAR_CONCAT"
    loose_dir.mkdir()
    other = _frame(5, 8)
    other.tofile(loose_dir / "0.pcd.bin")

    # The loose file wins when it exists, the pack serves the frame otherwise.
    load = t4pack.load_point_cloud_file
    assert np.array_equal(load(str(loose_dir / "0.pcd.bin"), 5), other)
    assert np.array_equal(load(str(loose_dir / "1.pcd.bin"), 5), frames[1])
    with pytest.raises(FileNotFoundError, match="nor its pack"):
        load(str(tmp_path / "db" / "scene" / "0" / "data" / "LIDAR_TOP" / "0.pcd.bin"), 5)
