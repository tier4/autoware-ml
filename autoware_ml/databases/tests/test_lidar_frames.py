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

"""Tests for the blob paths a lidar frame record carries."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.schemas.utils import relative_to_database_root

_SCENE = "db_default/0a1b2c3d-0000-4000-8000-000000000001/0"
_POINTCLOUD_TAIL = "data/LIDAR_CONCAT/00000.pcd.bin"
_MASK_TAIL = "lidarseg_auto/annotation/0123456789abcdef0123456789abcdef.bin"


def build_lidar_frame(
    pointcloud_path: str,
    source_path: str | None = None,
    semantic_mask_path: str | None = None,
) -> LidarFrameDataModel:
    """
    Lidar frame carrying the given blob paths and placeholder geometry.

    Args:
      pointcloud_path: Stored pointcloud path.
      source_path: Stored pointcloud source path.
      semantic_mask_path: Stored semantic mask path.

    Returns:
      LidarFrameDataModel: Lidar frame record.
    """
    return LidarFrameDataModel(
        lidar_frame_id="token",
        lidar_keyframe=True,
        lidar_sensor_id="sensor",
        lidar_sensor_channel_name="LIDAR_CONCAT",
        lidar_timestamp_seconds=0.0,
        lidar_pointcloud_relative_path=pointcloud_path,
        lidar_pointcloud_source_relative_path=source_path,
        lidar_pointcloud_num_features=5,
        lidar_sensor_to_ego_pose_matrix=np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
        lidar_pointcloud_semantic_mask_relative_path=semantic_mask_path,
    )


def test_relative_paths_round_trip_through_the_record_table() -> None:
    frame = build_lidar_frame(
        pointcloud_path=f"{_SCENE}/{_POINTCLOUD_TAIL}",
        source_path=f"{_SCENE}/{_POINTCLOUD_TAIL}",
        semantic_mask_path=f"{_SCENE}/{_MASK_TAIL}",
    )

    restored = LidarFrameDataModel.load_from_dictionary(frame.to_dictionary())

    assert restored.lidar_pointcloud_relative_path == f"{_SCENE}/{_POINTCLOUD_TAIL}"
    assert restored.lidar_pointcloud_source_relative_path == f"{_SCENE}/{_POINTCLOUD_TAIL}"
    assert restored.lidar_pointcloud_semantic_mask_relative_path == f"{_SCENE}/{_MASK_TAIL}"


def test_optional_paths_stay_absent() -> None:
    frame = build_lidar_frame(pointcloud_path=f"{_SCENE}/{_POINTCLOUD_TAIL}")
    assert frame.lidar_pointcloud_source_relative_path is None
    assert frame.lidar_pointcloud_semantic_mask_relative_path is None


@pytest.mark.parametrize(
    "field_name, paths",
    [
        (
            "lidar_pointcloud_relative_path",
            {"pointcloud_path": f"/data/t4dataset/{_SCENE}/{_POINTCLOUD_TAIL}"},
        ),
        (
            "lidar_pointcloud_source_relative_path",
            {
                "pointcloud_path": f"{_SCENE}/{_POINTCLOUD_TAIL}",
                "source_path": f"/data/t4dataset/{_SCENE}/{_POINTCLOUD_TAIL}",
            },
        ),
        (
            "lidar_pointcloud_semantic_mask_relative_path",
            {
                "pointcloud_path": f"{_SCENE}/{_POINTCLOUD_TAIL}",
                "semantic_mask_path": f"/data/t4dataset/{_SCENE}/{_MASK_TAIL}",
            },
        ),
    ],
)
def test_an_absolute_path_is_rejected(field_name: str, paths: dict[str, str]) -> None:
    # A path stored with the root of the generating machine is only valid on that machine, so
    # the record is rejected instead of binding the cache to one root
    with pytest.raises(ValueError, match=f"{field_name} .* is absolute"):
        build_lidar_frame(**paths)


def test_an_empty_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="lidar_pointcloud_relative_path is empty"):
        build_lidar_frame(pointcloud_path="")


def test_the_generators_strip_their_own_root() -> None:
    # Whatever root a generator ran under, the stored path starts below it, so a record keeps
    # the scene directory and resolves under any root the database is mounted at later
    for root in ("/data/t4dataset", "/mnt/other_host/data/t4dataset"):
        assert (
            relative_to_database_root(f"{root}/{_SCENE}/{_POINTCLOUD_TAIL}", root)
            == f"{_SCENE}/{_POINTCLOUD_TAIL}"
        )


def test_a_file_outside_the_root_cannot_be_stored() -> None:
    with pytest.raises(ValueError, match="does not lie under the database root"):
        relative_to_database_root("/elsewhere/samples/LIDAR_TOP/x.pcd.bin", "/data/nuscenes")
