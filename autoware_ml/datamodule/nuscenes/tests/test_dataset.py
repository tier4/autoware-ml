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

"""Tests for the samples the nuScenes dataset serves from a nuScenes record."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import polars as pl
import pytest

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDataModel
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.datamodule.nuscenes.dataset import NuScenesDataset
from autoware_ml.types.geometry import Box3DFieldIndex

_SCENE_NAME = "scene-0061"
_SAMPLE_LIDAR = (
    "samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
)
_SWEEP_LIDAR = "sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin"
_SAMPLE_IMAGE = "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"


def lidar_frame(frame_id: str, relative_path: str, keyframe: bool) -> dict:
    """
    Record table entry of one lidar frame, stored as the nuScenes generator stores it.

    Args:
      frame_id: Sample data token of the frame.
      relative_path: Path of the frame relative to the database root.
      keyframe: Whether the frame is the sample frame.

    Returns:
      dict: The lidar frame as the record table stores it.
    """
    return LidarFrameDataModel(
        lidar_frame_id=frame_id,
        lidar_keyframe=keyframe,
        lidar_sensor_id="lidar-sensor",
        lidar_sensor_channel_name="LIDAR_TOP",
        lidar_timestamp_seconds=1533151603.5 if keyframe else 1533151603.55,
        lidar_pointcloud_relative_path=relative_path,
        lidar_pointcloud_source_relative_path=None,
        lidar_pointcloud_num_features=5,
        lidar_sensor_to_ego_pose_matrix=np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
        lidar_pointcloud_semantic_mask_relative_path=None,
    ).to_dictionary()


def image_frame(relative_path: str) -> dict:
    """
    Record table entry of one camera keyframe, stored as the nuScenes generator stores it.

    Args:
      relative_path: Path of the image relative to the database root.

    Returns:
      dict: The image frame as the record table stores it.
    """
    return ImageFrameDataModel(
        image_frame_id="cam-front-token",
        image_keyframe=True,
        image_sensor_id="cam-front-sensor",
        image_sensor_channel_name="CAM_FRONT",
        image_timestamp_seconds=1533151603.5,
        image_relative_path=relative_path,
        image_height=900,
        image_width=1600,
        cam2img=np.eye(3),
        # nuScenes images ship rectified
        image_distortion_coefficients=[],
        image_distortion_model="",
        image_sensor_to_ego_pose_matrix=np.eye(4),
        image_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar2cam=np.eye(4),
        lidar2img=np.eye(4),
    ).to_dictionary()


def nuscenes_records() -> pl.DataFrame:
    """One nuScenes record: a sample frame, one sweep, one camera and one car."""
    box = np.zeros(len(Box3DFieldIndex), dtype=np.float64)
    box[Box3DFieldIndex.LENGTH] = 4.0
    box[Box3DFieldIndex.WIDTH] = 2.0
    box[Box3DFieldIndex.HEIGHT] = 1.5
    row = {
        DatasetTableSchema.SCENARIO_ID.name: _SCENE_NAME,
        DatasetTableSchema.SAMPLE_ID.name: "sample-token",
        DatasetTableSchema.SAMPLE_INDEX.name: 0,
        DatasetTableSchema.TIMESTAMP_SECONDS.name: 1533151603.5,
        DatasetTableSchema.LOCATION.name: None,
        DatasetTableSchema.VEHICLE_TYPE.name: None,
        DatasetTableSchema.SCENARIO_NAME.name: _SCENE_NAME,
        DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name: True,
        DatasetTableSchema.LIDAR_FRAMES.name: [
            lidar_frame("lidar-sample-token", _SAMPLE_LIDAR, keyframe=True),
            lidar_frame("lidar-sweep-token", _SWEEP_LIDAR, keyframe=False),
        ],
        DatasetTableSchema.LIDAR_SOURCES.name: [],
        DatasetTableSchema.IMAGE_FRAMES.name: [[image_frame(_SAMPLE_IMAGE)]],
        DatasetTableSchema.CATEGORY_MAPPING.name: None,
        DatasetTableSchema.BOXES_3D.name: [
            {
                "box3d_params": box.tolist(),
                "box3d_instance_id": "car-instance",
                "box3d_dataset_label_name": "vehicle.car",
                "box3d_label_name": "car",
                "box3d_label_index": 0,
                "box3d_num_lidar_points": 64,
                "box3d_num_radar_points": 0,
                "box3d_valid": True,
                "box3d_attributes": [],
                "box3d_coordinate": "gravity_center",
            }
        ],
    }
    return pl.DataFrame([row], schema=DatasetTableSchema.to_polars_schema())


def build_dataset(database_root_path: str) -> NuScenesDataset:
    """Dataset over the one nuScenes record, serving the 3D boxes."""
    return NuScenesDataset(
        database_root_path=database_root_path,
        max_num_3d_gt_bboxes=8,
        dataset_records_dataframe=nuscenes_records(),
        transforms=None,
        dataset_tasks=MappingProxyType({"detection3d": Detection3DTask}),
        lidar_intensity_scale=255.0,
    )


@pytest.mark.parametrize("root", ["/data/nuscenes", "/mnt/other_host/data/nuscenes"])
def test_a_nuscenes_record_is_served_under_the_root_of_the_dataset(root: str) -> None:
    """
    Input: a nuScenes record as the generator stores it, with its frame paths relative to the
        root and its scene name as scenario id, read by datasets over roots of different depths.
    Expected: the sample resolves every frame under the root of the dataset reading it, and its
        frame metadata names the scene of the record, which the flat nuScenes layout cannot be
        read off a frame path. nuScenes ships no lanelet maps, so the frame carries none.
    Check: the frame paths, the scene token, the map path and the boxes of the sample.
    """
    sample = build_dataset(root).get_data_sample(0)

    assert sample.lidar_point_cloud_samples is not None
    assert [frame.point_cloud_path for frame in sample.lidar_point_cloud_samples] == [
        f"{root}/{_SAMPLE_LIDAR}",
        f"{root}/{_SWEEP_LIDAR}",
    ]
    assert sample.image_samples is not None
    assert [image.image_path for image in sample.image_samples] == [f"{root}/{_SAMPLE_IMAGE}"]
    assert sample.frame_meta.scene_token == _SCENE_NAME
    assert sample.frame_meta.lanelet_map_path is None
    assert sample.detection3d_gt_bboxes_3d is not None
    assert len(sample.detection3d_gt_bboxes_3d) == 1


def test_the_frame_meta_composes_the_lidar_mounting_into_the_map_transform() -> None:
    """
    Input: the record read without a lidar source view.
    Expected: the map transform of the frame is the ego pose of the frame composed with the
        lidar mounting, both the identity here, so the sample frame maps onto itself.
    Check: the transform.
    """
    frame_meta = build_dataset("/data/nuscenes").get_frame_meta_sample(0, None)

    assert frame_meta.scene_token == _SCENE_NAME
    assert np.allclose(frame_meta.ego2global.numpy(), np.eye(4))
