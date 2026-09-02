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

"""Cross-namespace tests for the geometric 3D augmentations.

The point_cloud / camera_lidar / camera variants share their math via
``transforms.geometry3d``. These tests assert that, for the same RNG draw and
inputs, the shared quantities come out identical across namespaces, and that
each variant fails loudly (no silent fallback) when its required keys are
missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.transforms import geometry3d as g3d
from autoware_ml.transforms.camera import geometry as cam
from autoware_ml.transforms.camera_lidar import geometry as cam_lidar
from autoware_ml.transforms.point_cloud import geometry as pc


def _sample() -> dict:
    return {
        "points": np.array([[1.0, 2.0, 0.5, 1.0], [-3.0, 4.0, -1.0, 0.5]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.3, 1.5, -0.5]], dtype=np.float32),
        "lidar2cam": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "camera_intrinsics": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }


def test_global_rot_scale_trans_scales_velocity_with_space() -> None:
    sample = {
        "points": np.array([[1.0, 2.0, 0.5, 1.0]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.3, 1.5, -0.5]], dtype=np.float32),
    }

    out = pc.GlobalRotScaleTrans(rot_range=[0.0, 0.0], scale_ratio_range=[0.5, 0.5])(sample)

    box = out["gt_boxes"][0]
    assert np.allclose(box[:3], [0.5, 1.0, 0.0])
    assert np.allclose(box[3:6], [2.0, 1.0, 0.5])
    assert np.isclose(box[6], 0.3)
    assert np.allclose(box[7:9], [0.75, -0.25])


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_global_rot_scale_trans_shared_math_matches_across_namespaces(seed: int) -> None:
    kwargs = dict(
        rot_range=[-0.5, 0.5], scale_ratio_range=[0.9, 1.1], translation_std=[0.5, 0.5, 0.2]
    )

    np.random.seed(seed)
    out_pc = pc.GlobalRotScaleTrans(**kwargs)(_sample())
    np.random.seed(seed)
    out_cl = cam_lidar.GlobalRotScaleTrans(**kwargs)(_sample())
    np.random.seed(seed)
    out_cam = cam.GlobalRotScaleTrans(**kwargs)(_sample())

    # Boxes are transformed identically by all three namespaces.
    assert np.allclose(out_pc["gt_boxes"], out_cl["gt_boxes"])
    assert np.allclose(out_pc["gt_boxes"], out_cam["gt_boxes"])
    # point_cloud and camera_lidar transform points identically.
    assert np.allclose(out_pc["points"], out_cl["points"])
    # camera-aware namespaces update the camera matrices identically.
    assert np.allclose(out_cl["lidar2cam"], out_cam["lidar2cam"])
    assert np.allclose(out_cl["lidar2img"], out_cam["lidar2img"])
    # No fallback: the lidar-only variant never touches camera matrices.
    assert np.allclose(out_pc["lidar2cam"], np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))
    # camera-only variant never transforms points (leaves them untouched).
    assert np.allclose(out_cam["points"], _sample()["points"])


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_random_flip_shared_math_matches_across_namespaces(seed: int) -> None:
    kwargs = dict(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=1.0)

    np.random.seed(seed)
    out_pc = pc.RandomFlip3D(**kwargs)(_sample())
    np.random.seed(seed)
    out_cl = cam_lidar.RandomFlip3D(**kwargs)(_sample())
    np.random.seed(seed)
    out_cam = cam.RandomFlip3D(**kwargs)(_sample())

    assert np.allclose(out_pc["gt_boxes"], out_cl["gt_boxes"])
    assert np.allclose(out_pc["gt_boxes"], out_cam["gt_boxes"])
    assert np.allclose(out_pc["points"], out_cl["points"])
    assert np.allclose(out_cl["bev_flip_matrix"], out_cam["bev_flip_matrix"])
    assert np.allclose(out_cl["lidar2cam"], out_cam["lidar2cam"])
    # camera-only variant never transforms points (leaves them untouched).
    assert np.allclose(out_cam["points"], _sample()["points"])


def _ego_pose() -> np.ndarray:
    # Non-trivial lidar->global pose: yaw + translation.
    yaw = 0.7
    pose = np.eye(4, dtype=np.float64)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[:3, 3] = [10.0, -4.0, 1.2]
    return pose


@pytest.mark.parametrize("transform_cls", ["rot_scale_trans", "flip"])
def test_camera_transforms_fold_augmentation_into_ego_poses(transform_cls: str) -> None:
    """After augmentation, ego_pose must map *augmented* lidar coords to the
    same global point as before (StreamPETR warps temporal memory through it)."""
    sample = _sample()
    sample["ego_pose"] = _ego_pose()
    sample["ego_pose_inv"] = np.linalg.inv(sample["ego_pose"])
    point_lidar = np.array([3.0, -2.0, 0.5, 1.0])
    point_global = sample["ego_pose"] @ point_lidar

    np.random.seed(3)
    if transform_cls == "rot_scale_trans":
        out = cam.GlobalRotScaleTrans(
            rot_range=[-0.5, 0.5], scale_ratio_range=[0.9, 1.1], translation_std=[0.5, 0.5, 0.2]
        )(sample)
        augmentation = out["global_aug_matrix"]
    else:
        out = cam.RandomFlip3D(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=1.0)(sample)
        augmentation = out["bev_flip_matrix"]

    point_augmented = augmentation @ point_lidar
    assert np.allclose(out["ego_pose"] @ point_augmented, point_global, atol=1e-5)
    assert np.allclose(out["ego_pose_inv"] @ point_global, point_augmented, atol=1e-5)
    # The pair stays mutually inverse.
    assert np.allclose(out["ego_pose"] @ out["ego_pose_inv"], np.eye(4), atol=1e-5)


def test_camera_and_camera_lidar_fold_ego_poses_identically() -> None:
    def _sample_with_pose() -> dict:
        sample = _sample()
        sample["ego_pose"] = _ego_pose()
        sample["ego_pose_inv"] = np.linalg.inv(sample["ego_pose"])
        return sample

    kwargs = dict(rot_range=[-0.5, 0.5], scale_ratio_range=[0.9, 1.1])
    np.random.seed(11)
    out_cam = cam.GlobalRotScaleTrans(**kwargs)(_sample_with_pose())
    np.random.seed(11)
    out_cl = cam_lidar.GlobalRotScaleTrans(**kwargs)(_sample_with_pose())
    assert np.allclose(out_cam["ego_pose"], out_cl["ego_pose"])
    assert np.allclose(out_cam["ego_pose_inv"], out_cl["ego_pose_inv"])


def test_transform_boxes_scales_velocity_with_the_world() -> None:
    input_dict = {
        "gt_boxes": np.array([[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.3, 1.5, -0.5]], dtype=np.float32)
    }
    identity = np.eye(3, dtype=np.float32)
    g3d.transform_boxes(input_dict, identity, 0.0, 2.0, np.zeros((1, 3), dtype=np.float32))
    assert np.allclose(input_dict["gt_boxes"][0, 7:9], [3.0, -1.0])


def test_point_cloud_requires_a_point_representation() -> None:
    with pytest.raises(KeyError):
        pc.GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.0, 1.0])(
            {"gt_boxes": np.zeros((1, 9), dtype=np.float32)}
        )


def test_camera_lidar_requires_camera_matrix() -> None:
    # lidar2cam missing -> loud failure, not a silent skip of the camera update.
    with pytest.raises(KeyError):
        cam_lidar.GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.0, 1.0])(
            {"points": np.zeros((1, 4), dtype=np.float32)}
        )


def test_camera_lidar_requires_a_point_representation() -> None:
    with pytest.raises(KeyError):
        cam_lidar.GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.0, 1.0])(
            {"lidar2cam": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))}
        )


def test_camera_requires_camera_matrix() -> None:
    with pytest.raises(KeyError):
        cam.GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.0, 1.0])(
            {"gt_boxes": np.zeros((1, 9), dtype=np.float32)}
        )
