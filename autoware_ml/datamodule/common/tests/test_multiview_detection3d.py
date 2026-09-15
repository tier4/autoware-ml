"""Tests for shared multiview detection3d dataset utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from autoware_ml.datamodule.base import DataModule
from autoware_ml.datamodule.common.multiview_detection3d import MultiviewDetection3DDataset
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


def test_multiview_detection_dataset_keeps_timestamp_float64(tmp_path: Path) -> None:
    """Epoch-second timestamps must survive batch collation at full precision.

    float32 has 256-second steps at 1.7e9 seconds, which silently zeroes the
    inter-frame deltas consumed by streaming temporal detectors.
    """
    ann_file = tmp_path / "infos.pkl"
    sample = {
        "token": "sample-1",
        "scene_token": "scene-1",
        "timestamp": 1740707698.147682,
        "lidar_points": {"lidar_path": "lidar-1.bin", "num_pts_feats": 5},
        "images": {},
        "instances": [],
    }
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": [sample], "metainfo": {"classes": ["car"]}}, file)

    dataset = _Dataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=[],
        filter_frames_with_camera_order=False,
    )

    timestamp = dataset.get_data_info(0)["timestamp"]
    coerced = DataModule._coerce_value(timestamp)

    assert isinstance(timestamp, np.float64)
    assert coerced.dtype == torch.float64
    assert float(coerced) == 1740707698.147682
