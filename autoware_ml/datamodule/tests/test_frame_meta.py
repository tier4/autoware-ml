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

"""Tests for the scene directory fragment the frame metadata is keyed by."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment


def test_scene_dir_fragment() -> None:
    path = "db_test_v2/0a1b2c3d-0000/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path) == "db_test_v2/0a1b2c3d-0000/0"


def test_scene_dir_fragment_rejects_absolute_paths() -> None:
    # The records store their paths relative to the database root, so a rooted path was not
    # written by this version of the generators
    with pytest.raises(ValueError, match="relative to the database root"):
        scene_dir_fragment("/data/t4/db_test_v2/0a1b2c3d-0000/0/data/LIDAR_CONCAT/00000.pcd.bin")


def test_scene_dir_fragment_rejects_short_paths() -> None:
    with pytest.raises(ValueError, match="scene directory"):
        scene_dir_fragment("no_scene.bin")
