"""Tests for shared multiview detection3d dataset utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from autoware_ml.datamodule.common.multiview_detection3d import MultiviewDetection3DDataset
from autoware_ml.datamodule.nuscenes.multiview_detection3d import (
    NuscenesMultiviewDetection3DDataset,
)
from autoware_ml.datamodule.t4dataset.multiview_detection3d import T4MultiviewDetection3DDataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.loading import LoadAnnotations3D
from autoware_ml.transforms.camera.loading import LoadMultiViewImagesFromFiles
from autoware_ml.transforms.point_cloud.sweeps import LoadPointsFromMultiSweeps


class _Dataset(MultiviewDetection3DDataset):
    pass


def test_multiview_detection_dataset_applies_loader_pipeline(tmp_path: Path) -> None:
    image_path = tmp_path / "cam.png"
    lidar_path = tmp_path / "lidar.bin"
    cv2.imwrite(str(image_path), np.full((4, 6, 3), 127, dtype=np.uint8))
    np.array([[1.0, 2.0, 3.0, 4.0, 9.0]], dtype=np.float32).tofile(lidar_path)

    ann_file = tmp_path / "infos.pkl"
    sample = {
        "token": "sample-1",
        "timestamp": 123,
        "scene_token": "scene-1",
        "ego2global_translation": [1.0, 2.0, 3.0],
        "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        "lidar_points": {"lidar_path": lidar_path.name, "num_pts_feats": 5},
        "images": {
            "CAM_FRONT": {
                "img_path": image_path.name,
                "cam2img": np.eye(3, dtype=np.float32),
                "lidar2cam": np.eye(4, dtype=np.float32),
            }
        },
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.1],
                "velocity": [0.2, -0.1],
                "bbox_label_3d": 0,
                "bbox_3d_isvalid": True,
                "num_lidar_pts": 5,
            }
        ],
    }
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": [sample], "metainfo": {"classes": ["car"]}}, file)

    dataset = _Dataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=["CAM_FRONT"],
        dataset_transforms=TransformsCompose(
            [
                LoadAnnotations3D(),
                LoadMultiViewImagesFromFiles(),
                LoadPointsFromMultiSweeps(load_dim=5, use_dim=[0, 1, 2, 3], sweeps_num=0),
            ]
        ),
    )

    output = dataset[0]

    assert output["points"].shape == (1, 4)
    assert output["img"].shape == (1, 3, 4, 6)
    assert output["camera_intrinsics"].shape == (1, 4, 4)
    assert output["lidar2cam"].shape == (1, 4, 4)
    assert output["lidar2img"].shape == (1, 4, 4)
    assert output["ego_pose"].shape == (4, 4)
    assert output["ego_pose_inv"].shape == (4, 4)
    assert output["scene_token"] == "scene-1"
    assert output["prev_exists"] == np.float32(0.0)
    assert output["gt_boxes"].shape == (1, 9)
    assert output["gt_labels"].tolist() == [0]


def test_nuscenes_multiview_exposes_the_ego_pose_from_either_record_form(
    tmp_path: Path,
) -> None:
    # NuScenes records carry the pose as translation plus rotation as well as a
    # ready matrix, and the metric-facing ego2global must come out of both.
    sample = {
        "token": "sample-1",
        "timestamp": 123,
        "scene_token": "scene-1",
        "ego2global_translation": [10.0, 20.0, 0.0],
        "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        "lidar_path": "lidar.bin",
        "lidar_points": {"lidar_path": "lidar.bin", "num_pts_feats": 5},
        "images": {
            "CAM_FRONT": {
                "img_path": "cam.png",
                "cam2img": np.eye(3, dtype=np.float32),
                "lidar2cam": np.eye(4, dtype=np.float32),
            }
        },
        "instances": [],
    }
    ann_file = tmp_path / "infos.pkl"
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": [sample], "metainfo": {"classes": ["car"]}}, file)

    dataset = NuscenesMultiviewDetection3DDataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=["CAM_FRONT"],
    )

    info = dataset.get_data_info(0)

    expected = np.eye(4)
    expected[:3, 3] = [10.0, 20.0, 0.0]
    assert np.allclose(info["ego2global"], expected)
    assert info["ego2global"].dtype == np.float64


def test_t4_multiview_exposes_metric_frame_context(tmp_path: Path) -> None:
    # The metric suites need the map-frame ego pose and the map-resolvable
    # <db>/<uuid>/<version> scene token, not the opaque annotation token.
    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, 3] = [10.0, 20.0, 0.0]
    sample = {
        "token": "sample-1",
        "timestamp": 123,
        "scene_token": "opaque-annotation-token",
        "ego2global": ego2global,
        "lidar_points": {
            "lidar_path": "db_x/uuid-1/0/data/LIDAR_CONCAT/0.pcd.bin",
            "num_pts_feats": 5,
        },
        "images": {
            "CAM_FRONT": {
                "img_path": "cam.png",
                "cam2img": np.eye(3, dtype=np.float32),
                "lidar2cam": np.eye(4, dtype=np.float32),
            }
        },
        "instances": [],
    }
    ann_file = tmp_path / "infos.pkl"
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": [sample], "metainfo": {"classes": ["car"]}}, file)

    dataset = T4MultiviewDetection3DDataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=["CAM_FRONT"],
    )

    info = dataset.get_data_info(0)

    assert info["scene_token"] == "db_x/uuid-1/0"
    assert np.allclose(info["ego2global"], ego2global)


def test_multiview_detection_dataset_builds_prev_exists_from_scene_tokens(
    tmp_path: Path,
) -> None:
    ann_file = tmp_path / "infos.pkl"
    samples = [
        {
            "token": "sample-1",
            "scene_token": "scene-1",
            "prev_exists": True,
            "lidar_points": {"lidar_path": "lidar-1.bin", "num_pts_feats": 5},
            "images": {},
            "instances": [],
        },
        {
            "token": "sample-2",
            "scene_token": "scene-1",
            "prev_exists": False,
            "lidar_points": {"lidar_path": "lidar-2.bin", "num_pts_feats": 5},
            "images": {},
            "instances": [],
        },
        {
            "token": "sample-3",
            "scene_token": "scene-2",
            "prev_exists": True,
            "lidar_points": {"lidar_path": "lidar-3.bin", "num_pts_feats": 5},
            "images": {},
            "instances": [],
        },
    ]
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": samples, "metainfo": {"classes": ["car"]}}, file)

    dataset = _Dataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=[],
        filter_frames_with_camera_order=False,
    )

    assert dataset.get_data_info(0)["prev_exists"] == np.float32(0.0)
    assert dataset.get_data_info(1)["prev_exists"] == np.float32(1.0)
    assert dataset.get_data_info(2)["prev_exists"] == np.float32(0.0)
