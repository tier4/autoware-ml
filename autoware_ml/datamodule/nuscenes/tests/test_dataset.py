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

"""Unit tests for the frame paths of the nuScenes dataset."""

from __future__ import annotations

import unittest

from autoware_ml.datamodule.nuscenes.dataset import NuScenesDataset


class TestNuScenesDatasetFramePaths(unittest.TestCase):
    """Stored root-relative frame paths are joined onto the database root of the dataset."""

    def _build_dataset(self, database_root_path: str) -> NuScenesDataset:
        return NuScenesDataset(
            database_root_path=database_root_path,
            transforms=None,
            dataset_tasks={},
            lidar_intensity_scale=255.0,
        )

    def test_joins_the_stored_path_onto_the_root(self) -> None:
        """
        Input: a sample, a sweep and a camera path as the records store them, relative to the
            root, and two datasets over roots of different depths.
        Expected: every path resolves under the root of the dataset reading it, so a cache
            built under one root serves another.
        Check: the resolved paths.
        """
        stored_paths = (
            "samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
            "sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin",
            "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
        )
        for root in ("/data/nuscenes", "/workspace/data/sets/nuscenes"):
            dataset = self._build_dataset(root)
            for stored_path in stored_paths:
                with self.subTest(root=root, path=stored_path):
                    self.assertEqual(
                        dataset._update_frame_path(stored_path), f"{root}/{stored_path}"
                    )


if __name__ == "__main__":
    unittest.main()
