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

"""Unit tests for the t4pack frame locations in the record table."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.lidar_frames import (
    LidarFrameDataModel,
    LidarFrameDatasetSchema,
)
from autoware_ml.databases.t4pack.t4pack_frame import T4PackFrame
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.databases.t4pack.tests.t4pack_fixtures import random_lidar_frame, write_test_pack

_T4PACK_FRAME = T4PackFrame(
    offset=8, size=1234, num_points=100, dtypes=("f4s", "f4s", "f4s", "u1", "i1")
)


def _lidar_frame(t4pack_frame: T4PackFrame | None) -> LidarFrameDataModel:
    """Lidar frame record with the given pack location."""
    return LidarFrameDataModel(
        lidar_frame_id="token",
        lidar_keyframe=True,
        lidar_sensor_id="sensor",
        lidar_sensor_channel_name="LIDAR_CONCAT",
        lidar_timestamp_seconds=1.0,
        lidar_pointcloud_path="/db/scene/0/data/LIDAR_CONCAT/00000.pcd.bin",
        lidar_pointcloud_source_path=None,
        lidar_pointcloud_num_features=5,
        lidar_sensor_to_ego_pose_matrix=np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
        lidar_pointcloud_semantic_mask_path=None,
        lidar_pointcloud_t4pack_frame=t4pack_frame,
    )


class TestT4PackFrameRecord(unittest.TestCase):
    """Unit tests for the t4pack frame column of the lidar frame records."""

    def test_t4pack_frame_survives_the_record_table(self) -> None:
        """Test that the location goes through the Parquet struct column and back unchanged."""
        column = LidarFrameDatasetSchema.lidar_pointcloud_t4pack_frame
        for t4pack_frame in (_T4PACK_FRAME, None):
            with self.subTest(t4pack_frame=t4pack_frame):
                stored = _lidar_frame(t4pack_frame).to_dictionary()[column.name]

                table = pl.DataFrame({column.name: [stored]}, schema={column.name: column.dtype})

                self.assertEqual(
                    LidarFrameDataModel.load_t4pack_frame(table.to_dicts()[0][column.name]),
                    t4pack_frame,
                )
                self.assertEqual(
                    LidarFrameDataModel.load_from_dictionary(
                        _lidar_frame(t4pack_frame).to_dictionary()
                    ).lidar_pointcloud_t4pack_frame,
                    t4pack_frame,
                )


class TestT4RecordsGeneratorT4PackFrame(unittest.TestCase):
    """Unit tests for the pack lookup of T4RecordsGenerator."""

    def setUp(self) -> None:
        """Create a scene directory and a generator with only the pack lookup state."""
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.data_path = Path(temporary_directory.name) / "data"
        self.data_path.mkdir()
        # Skip loading a T4 scene, the pack lookup does not need one
        self.generator = T4RecordsGenerator.__new__(T4RecordsGenerator)
        self.generator._t4pack_indices = {}

    def _write_pack(self) -> None:
        """Write a channel pack holding the frame ``00000.pcd.bin`` of 7 points."""
        write_test_pack(
            self.data_path / "LIDAR_CONCAT.pack",
            {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(0), 7)},
        )

    def test_generator_records_the_location_of_a_packed_frame(self) -> None:
        """Test that a frame of a scene with a channel pack gets the location from its index."""
        self._write_pack()

        t4pack_frame = self.generator._find_t4pack_frame(
            str(self.data_path / "LIDAR_CONCAT" / "00000.pcd.bin")
        )

        self.assertIsNotNone(t4pack_frame)
        self.assertEqual(t4pack_frame.num_points, 7)

    def test_generator_records_no_location_without_a_pack(self) -> None:
        """Test that scenes with loose frames only keep an empty location."""
        path = self.data_path / "LIDAR_CONCAT" / "00000.pcd.bin"

        self.assertIsNone(self.generator._find_t4pack_frame(str(path)))

    def test_generator_rejects_a_frame_missing_from_the_pack(self) -> None:
        """Test that a pack without a frame of the scene fails record generation loudly."""
        self._write_pack()

        with self.assertRaisesRegex(ValueError, "has no frame 00001.pcd.bin"):
            self.generator._find_t4pack_frame(
                str(self.data_path / "LIDAR_CONCAT" / "00001.pcd.bin")
            )


if __name__ == "__main__":
    unittest.main()
