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

"""Tests for the t4pack frame locations in the record table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from autoware_ml.databases.schemas.lidar_frames import (
    LidarFrameDatasetSchema,
    LidarFrameDataModel,
    load_t4pack_frame_location,
)
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.utils.point_cloud.t4pack import T4PackFrameLocation
from autoware_ml.utils.tests.t4pack_fixtures import random_lidar_frame, write_test_pack

LOCATION = T4PackFrameLocation(
    offset=8, size=1234, num_points=100, dtypes=("f4s", "f4s", "f4s", "u1", "i1")
)


def _lidar_frame(location: T4PackFrameLocation | None) -> LidarFrameDataModel:
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
        lidar_pointcloud_t4pack_frame=location,
    )


@pytest.mark.parametrize("location", [LOCATION, None])
def test_location_survives_the_record_table(location: T4PackFrameLocation | None) -> None:
    """The location goes through the Parquet struct column and back unchanged."""
    column = LidarFrameDatasetSchema.lidar_pointcloud_t4pack_frame
    stored = _lidar_frame(location).to_dictionary()[column.name]

    table = pl.DataFrame({column.name: [stored]}, schema={column.name: column.dtype})

    assert load_t4pack_frame_location(table.to_dicts()[0][column.name]) == location
    assert LidarFrameDataModel.load_from_dictionary(
        _lidar_frame(location).to_dictionary()
    ).lidar_pointcloud_t4pack_frame == (location)


def _generator() -> T4RecordsGenerator:
    """A generator with only the pack lookup state, without loading a T4 scene."""
    generator = T4RecordsGenerator.__new__(T4RecordsGenerator)
    generator._t4pack_indices = {}
    return generator


def test_generator_records_the_location_of_a_packed_frame(tmp_path: Path) -> None:
    """A frame of a scene with a channel pack gets the location the pack index gives it."""
    (tmp_path / "data").mkdir()
    write_test_pack(
        tmp_path / "data" / "LIDAR_CONCAT.pack",
        {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(0), 7)},
    )

    location = _generator()._find_t4pack_frame(
        str(tmp_path / "data" / "LIDAR_CONCAT" / "00000.pcd.bin")
    )

    assert location is not None and location.num_points == 7


def test_generator_records_no_location_without_a_pack(tmp_path: Path) -> None:
    """Scenes with loose frames only keep an empty location."""
    path = tmp_path / "data" / "LIDAR_CONCAT" / "00000.pcd.bin"

    assert _generator()._find_t4pack_frame(str(path)) is None


def test_generator_rejects_a_frame_missing_from_the_pack(tmp_path: Path) -> None:
    """A pack that does not hold a frame of the scene fails record generation loudly."""
    (tmp_path / "data").mkdir()
    write_test_pack(
        tmp_path / "data" / "LIDAR_CONCAT.pack",
        {"00000.pcd.bin": random_lidar_frame(np.random.default_rng(0), 7)},
    )

    with pytest.raises(ValueError, match="has no frame 00001.pcd.bin"):
        _generator()._find_t4pack_frame(str(tmp_path / "data" / "LIDAR_CONCAT" / "00001.pcd.bin"))
