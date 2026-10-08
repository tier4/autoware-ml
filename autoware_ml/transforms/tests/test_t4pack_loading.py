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

"""Unit tests for the point cloud file formats of the multi-task point cloud loaders."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from autoware_ml.databases.t4pack.t4pack_frame import T4PackFrame
from autoware_ml.databases.t4pack.t4pack import T4Pack
from autoware_ml.databases.t4pack.tests.t4pack_fixtures import random_lidar_frame, write_test_pack
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.transforms.point_cloud.loading import (
    LoadMultiSweepPointsFromFile,
    LoadPointsFromFile,
    SweepWindow,
)
from autoware_ml.types.dataset import PCDFileFormat

_NAMES = ("00000.pcd.bin", "00001.pcd.bin", "00002.pcd.bin")


def _record(path: Path, i: int, t4pack_frame: T4PackFrame | None) -> LiDARPointCloudSample:
    """Point cloud record of the ``i``-th frame, 0.1 s older than the previous one."""
    return LiDARPointCloudSample(
        point_cloud_path=str(path),
        timestamp=10.0 - 0.1 * i,
        num_features=5,
        intensity_scale=1.0,
        sensor_to_ego_pose_matrix=torch.eye(4),
        lidar_to_ego_pose_to_global_matrix=torch.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=torch.eye(4),
        t4pack_frame=t4pack_frame,
    )


def _load(
    records: list[LiDARPointCloudSample], pcd_file_format: PCDFileFormat, sweeps_num: int = 0
) -> torch.Tensor:
    """Load the current frame, and its sweeps when ``sweeps_num`` is set."""
    sample = ModelGTSample(
        lidar_point_cloud_samples=records,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )
    sample = LoadPointsFromFile(
        use_dim=[0, 1, 2, 3], bev_remove_radius=1.0, pcd_file_format=pcd_file_format
    )(sample)
    if sweeps_num:
        sample = LoadMultiSweepPointsFromFile(
            past=SweepWindow(num=sweeps_num, time_lag_range=(0.05, 0.25), selection="nearest"),
            use_dim=[0, 1, 2, 3],
            pcd_file_format=pcd_file_format,
        )(sample)
    return sample.point_cloud_data.points


class TestPCDFileFormat(unittest.TestCase):
    """Unit tests for the pcd_file_format of LoadPointsFromFile and LoadMultiSweepPointsFromFile."""

    def setUp(self) -> None:
        """Write the same three frames as loose files and as a pack, with their records."""
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        rng = np.random.default_rng(0)
        frames = {name: random_lidar_frame(rng, 50 + 10 * i) for i, name in enumerate(_NAMES)}

        loose_path = root / "loose" / "data" / "LIDAR_CONCAT"
        loose_path.mkdir(parents=True)
        for name, frame in frames.items():
            frame.tofile(loose_path / name)
        packed_path = root / "packed" / "data"
        packed_path.mkdir(parents=True)
        write_test_pack(packed_path / "LIDAR_CONCAT.pack", frames)
        index = T4Pack(str(packed_path / "LIDAR_CONCAT.pack")).read_index()

        # Records of a scene without a pack, and of a scene with only the pack
        self.loose = [_record(loose_path / name, i, None) for i, name in enumerate(_NAMES)]
        self.packed = [
            _record(packed_path / "LIDAR_CONCAT" / name, i, index[name])
            for i, name in enumerate(_NAMES)
        ]

    def test_pack_loads_the_same_points_as_the_bin_files(self) -> None:
        """Test that the frame and its sweeps load the same points from the pack as from bin."""
        for pcd_file_format in (PCDFileFormat.T4PACK, PCDFileFormat.AUTO):
            for sweeps_num in (0, 2):
                with self.subTest(pcd_file_format=pcd_file_format, sweeps_num=sweeps_num):
                    expected = _load(self.loose, PCDFileFormat.BIN, sweeps_num)

                    actual = _load(self.packed, pcd_file_format, sweeps_num)

                    self.assertTrue(torch.equal(actual, expected))

    def test_auto_reads_the_pack_when_the_record_has_a_location(self) -> None:
        """Test that auto does not touch the ``.pcd.bin`` path of a record with a location."""
        # The packed records point at loose paths that do not exist
        self.assertFalse(Path(self.packed[0].point_cloud_path).exists())

        self.assertGreater(_load(self.packed, PCDFileFormat.AUTO).shape[0], 0)

    def test_auto_reads_the_bin_file_without_a_pack_location(self) -> None:
        """Test that records of scenes without a pack load from their ``.pcd.bin`` files."""
        self.assertTrue(
            torch.equal(_load(self.loose, PCDFileFormat.AUTO), _load(self.loose, PCDFileFormat.BIN))
        )

    def test_bin_ignores_the_pack_location(self) -> None:
        """Test that bin reads the ``.pcd.bin`` path even when the record has a pack location."""
        with self.assertRaises(FileNotFoundError):
            _load(self.packed, PCDFileFormat.BIN)

    def test_t4pack_rejects_a_record_without_a_pack_location(self) -> None:
        """Test that records of scenes without a pack name the formats to use instead."""
        with self.assertRaisesRegex(ValueError, "has no t4pack location"):
            _load(self.loose, PCDFileFormat.T4PACK)

    def test_a_string_format_is_rejected(self) -> None:
        """Test that the format must be a PCDFileFormat, not its string value."""
        with self.assertRaisesRegex(TypeError, "must be a PCDFileFormat"):
            LoadPointsFromFile(pcd_file_format="t4pack")


if __name__ == "__main__":
    unittest.main()
