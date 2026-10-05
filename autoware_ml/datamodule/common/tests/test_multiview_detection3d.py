"""Tests for shared multiview detection3d dataset utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from autoware_ml.datamodule.base import DataModule
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


def _nuscenes_multiview_sample(pose_fields: dict) -> dict:
    """A nuScenes multiview record carrying the given ego pose fields."""
    return {
        "token": "sample-1",
        "timestamp": 123,
        "scene_token": "scene-1",
        **pose_fields,
        "lidar_path": "lidar.bin",
        "lidar_points": {
            "lidar_path": "lidar.bin",
            "num_pts_feats": 5,
            "lidar2ego": np.eye(4),
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


def _nuscenes_multiview_ego2global(tmp_path: Path, sample: dict) -> np.ndarray:
    """The ego2global the dataset exposes for one record."""
    ann_file = tmp_path / "infos.pkl"
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": [sample], "metainfo": {"classes": ["car"]}}, file)

    dataset = NuscenesMultiviewDetection3DDataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=["CAM_FRONT"],
    )
    return dataset.get_data_info(0)["ego2global"]


def test_nuscenes_multiview_builds_the_ego_pose_from_translation_and_rotation(
    tmp_path: Path,
) -> None:
    sample = _nuscenes_multiview_sample(
        {
            "ego2global_translation": [10.0, 20.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        }
    )

    ego2global = _nuscenes_multiview_ego2global(tmp_path, sample)

    expected = np.eye(4)
    expected[:3, 3] = [10.0, 20.0, 0.0]
    assert np.allclose(ego2global, expected)
    assert ego2global.dtype == np.float64


def test_nuscenes_multiview_takes_the_ego_pose_matrix_as_stored(tmp_path: Path) -> None:
    pose = np.eye(4)
    pose[:3, 3] = [10.0, 20.0, 0.0]
    sample = _nuscenes_multiview_sample({"ego2global": pose})

    ego2global = _nuscenes_multiview_ego2global(tmp_path, sample)

    assert np.allclose(ego2global, pose)
    assert ego2global.dtype == np.float64


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


def _scene_dataset(tmp_path: Path, scene_tokens: list[str]) -> _Dataset:
    ann_file = tmp_path / "infos.pkl"
    samples = [
        {
            "token": f"sample-{index}",
            "scene_token": token,
            "lidar_points": {"lidar_path": f"lidar-{index}.bin", "num_pts_feats": 5},
            "images": {},
            "instances": [],
        }
        for index, token in enumerate(scene_tokens)
    ]
    with open(ann_file, "wb") as file:
        pickle.dump({"data_list": samples, "metainfo": {"classes": ["car"]}}, file)
    return _Dataset(
        data_root=str(tmp_path),
        ann_file=str(ann_file),
        class_names=["car"],
        camera_order=[],
        filter_frames_with_camera_order=False,
    )


def test_scene_index_groups_follow_file_order(tmp_path: Path) -> None:
    dataset = _scene_dataset(tmp_path, ["scene-1", "scene-1", "scene-2", "scene-3", "scene-3"])

    assert dataset.scene_index_groups() == [[0, 1], [2], [3, 4]]


def test_scene_index_groups_returns_a_fresh_copy(tmp_path: Path) -> None:
    dataset = _scene_dataset(tmp_path, ["scene-1", "scene-1", "scene-2"])

    dataset.scene_index_groups()[0].append(99)

    assert dataset.scene_index_groups() == [[0, 1], [2]]


def test_interleaved_scenes_load_but_fail_at_the_streaming_accessor(tmp_path: Path) -> None:
    # prev_exists is derived from file adjacency, so an interleaved file would
    # silently reset temporal memory mid-scene. Non-temporal multiview models
    # never call scene_index_groups(), so construction itself must still work.
    dataset = _scene_dataset(tmp_path, ["scene-1", "scene-2", "scene-1"])

    assert len(dataset) == 3
    with pytest.raises(ValueError, match="interleaves scene 'scene-1'"):
        dataset.scene_index_groups()
