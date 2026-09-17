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

_SCENE = "db_j6gen2_v4/00fa6989-549a-4ab2-b5ea-aa07f416d821/0"
_POINTCLOUD_TAIL = "data/LIDAR_CONCAT/00000.pcd.bin"
_MASK_TAIL = "lidarseg_auto/annotation/747de21bbb6b0e58f22e58231067fa5f.bin"


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
        lidar_pointcloud_path=pointcloud_path,
        lidar_pointcloud_source_path=source_path,
        lidar_pointcloud_num_features=5,
        lidar_sensor_to_ego_pose_matrix=np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
        lidar_pointcloud_semantic_mask_path=semantic_mask_path,
    )


def test_relative_paths_keep_the_scene_whatever_root_generated_them() -> None:
    frame = build_lidar_frame(
        pointcloud_path=f"/mnt/some_other_host/t4dataset/{_SCENE}/{_POINTCLOUD_TAIL}",
        source_path=f"/mnt/some_other_host/t4dataset/{_SCENE}/{_POINTCLOUD_TAIL}",
        semantic_mask_path=f"/workspace/data/t4dataset/pseudo_large/{_SCENE}/{_MASK_TAIL}",
    )
    # Only the part below the database root is kept. It includes the scene, so files with the
    # same name in two scenes stay apart.
    assert frame.lidar_pointcloud_relative_path == f"{_SCENE}/{_POINTCLOUD_TAIL}"
    assert frame.lidar_pointcloud_source_relative_path == f"{_SCENE}/{_POINTCLOUD_TAIL}"
    assert frame.lidarseg_pointcloud_semantic_mask_relative_path == f"{_SCENE}/{_MASK_TAIL}"


def test_optional_paths_stay_absent() -> None:
    frame = build_lidar_frame(
        pointcloud_path=f"/workspace/data/t4dataset/pseudo_large/{_SCENE}/{_POINTCLOUD_TAIL}"
    )
    assert frame.lidar_pointcloud_source_relative_path is None
    assert frame.lidarseg_pointcloud_semantic_mask_relative_path is None


def test_a_scene_relative_mask_path_is_rejected() -> None:
    # A mask path relative to the scene loses the scene, so the dataset would read the mask of
    # another scene. The record must be rejected.
    frame = build_lidar_frame(
        pointcloud_path=f"/workspace/data/t4dataset/pseudo_large/{_SCENE}/{_POINTCLOUD_TAIL}",
        semantic_mask_path=_MASK_TAIL,
    )
    with pytest.raises(ValueError, match="lidar_pointcloud_semantic_mask_path"):
        frame.lidarseg_pointcloud_semantic_mask_relative_path


def test_a_scene_relative_pointcloud_path_is_rejected() -> None:
    frame = build_lidar_frame(pointcloud_path=_POINTCLOUD_TAIL)
    with pytest.raises(ValueError, match="lidar_pointcloud_path"):
        frame.lidar_pointcloud_relative_path
