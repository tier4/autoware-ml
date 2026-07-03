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
