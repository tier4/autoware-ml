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

"""Unit tests for the t4pack reader."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from autoware_ml.databases.t4pack.t4pack_frame import T4PackFrame
from autoware_ml.databases.t4pack.t4pack import T4Pack
from autoware_ml.databases.t4pack.tests.t4pack_fixtures import random_lidar_frame, write_test_pack


class TestT4Pack(unittest.TestCase):
    """Unit tests for T4Pack."""

    def setUp(self) -> None:
        """Create a temporary directory for the packs of a test."""
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.pack_path = Path(temporary_directory.name) / "LIDAR_CONCAT.pack"

    def test_frames_read_back_bit_exact(self) -> None:
        """Test that every frame decodes to the float32 matrix that was packed, bit for bit."""
        rng = np.random.default_rng(0)
        frames = {f"{i:05d}.pcd.bin": random_lidar_frame(rng, 100 + i) for i in range(3)}
        write_test_pack(self.pack_path, frames)
        pack = T4Pack(str(self.pack_path))

        index = pack.read_index()

        self.assertEqual(list(index), list(frames))
        for name, frame in frames.items():
            points = pack.read_frame(index[name], 5)
            self.assertEqual(points.dtype, np.float32)
            np.testing.assert_array_equal(points.view(np.uint32), frame.view(np.uint32))

    def test_float_intensity_column_keeps_values_above_255(self) -> None:
        """Test that a float intensity column returns intensities a u8 column cannot hold."""
        rng = np.random.default_rng(1)
        frame = random_lidar_frame(rng, 64)
        frame[:, 3] = rng.uniform(0.0, 65535.0, 64).astype(np.float32)
        write_test_pack(
            self.pack_path, {"00000.pcd.bin": frame}, dtypes=("f4s", "f4s", "f4s", "f4s", "i1")
        )
        pack = T4Pack(str(self.pack_path))

        t4pack_frame = pack.read_index()["00000.pcd.bin"]

        np.testing.assert_array_equal(pack.read_frame(t4pack_frame, 5), frame)

    def test_index_keeps_the_file_names_of_the_scene(self) -> None:
        """Test that frames named without zero padding are indexed under their own names."""
        rng = np.random.default_rng(2)
        names = ["0.pcd.bin", "1.pcd.bin", "10.pcd.bin", "-1-01.pcd.bin"]
        frames = {name: random_lidar_frame(rng, 10 + i) for i, name in enumerate(names)}
        write_test_pack(self.pack_path, frames)
        pack = T4Pack(str(self.pack_path))

        index = pack.read_index()

        self.assertEqual(sorted(index), sorted(names))
        np.testing.assert_array_equal(pack.read_frame(index["10.pcd.bin"], 5), frames["10.pcd.bin"])

    def test_a_file_that_is_not_a_pack_is_rejected(self) -> None:
        """Test that the index reader checks the magic at both ends of the file."""
        self.pack_path.write_bytes(b"\x00" * 64)

        with self.assertRaisesRegex(ValueError, "not a t4pack file"):
            T4Pack(str(self.pack_path)).read_index()

    def test_a_frame_with_another_column_count_is_rejected(self) -> None:
        """Test that the record's feature count must match the columns of the packed frame."""
        write_test_pack(
            self.pack_path, {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(3), 8)}
        )
        pack = T4Pack(str(self.pack_path))
        t4pack_frame = pack.read_index()["00000.pcd.bin"]

        with self.assertRaisesRegex(ValueError, "the record declares 4 features"):
            pack.read_frame(t4pack_frame, 4)

    def test_a_frame_past_the_end_of_the_pack_is_rejected(self) -> None:
        """Test that a stale location, e.g. of a rewritten pack, does not read garbage."""
        write_test_pack(
            self.pack_path, {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(4), 8)}
        )
        t4pack_frame = T4PackFrame(
            offset=self.pack_path.stat().st_size - 4,
            size=64,
            num_points=8,
            dtypes=("f4s",) * 3 + ("u1", "i1"),
        )

        with self.assertRaisesRegex(ValueError, "ends before the frame"):
            T4Pack(str(self.pack_path)).read_frame(t4pack_frame, 5)

    def test_the_pack_of_a_frame_replaces_its_channel_directory(self) -> None:
        """Test that ``data/<channel>/<name>`` is held by ``data/<channel>.pack``."""
        pack = T4Pack.from_point_cloud_path("/db/scene/0/data/LIDAR_CONCAT/00012.pcd.bin")

        self.assertEqual(pack.pack_path, "/db/scene/0/data/LIDAR_CONCAT.pack")


if __name__ == "__main__":
    unittest.main()
