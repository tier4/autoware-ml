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

"""Unit tests for the Sparse4D lidar branch."""

from __future__ import annotations

import pytest
import torch.nn as nn

from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.detection3d.main_modules.sparse4ds.tests.lidar_fixtures import (
    BEV_SHAPE,
    MIDDLE_CHANNELS,
    NECK_CHANNELS,
    POINT_CHANNELS,
    StubMiddleEncoder,
    StubVoxelEncoder,
    build_lidar_network,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import build_inputs


def test_middle_encoder_must_be_a_bev_middle_encoder() -> None:
    with pytest.raises(TypeError, match="BEVMiddleEncoder"):
        Sparse4DLidar(
            pts_voxel_encoder=StubVoxelEncoder(POINT_CHANNELS, MIDDLE_CHANNELS),
            pts_middle_encoder=nn.Identity(),
            pts_backbone=nn.Identity(),
            pts_neck=nn.Identity(),
        )


def test_expected_bev_shape_is_the_one_of_the_middle_encoder() -> None:
    assert build_lidar_network().expected_bev_shape == BEV_SHAPE


def test_forward_returns_the_bev_map_of_the_neck() -> None:
    branch = build_lidar_network()
    voxels_data = build_inputs().voxels_data

    bev_map = branch(
        voxels=voxels_data.voxels,
        coords=voxels_data.batch_zyx_coords(),
        num_points=voxels_data.num_points,
        batch_size=1,
    )

    assert bev_map.shape == (1, NECK_CHANNELS, BEV_SHAPE[0], BEV_SHAPE[1])


def test_prepare_for_export_swaps_the_middle_encoder_in_place() -> None:
    branch = build_lidar_network()

    exportable = branch.prepare_for_export()

    assert isinstance(exportable, StubMiddleEncoder)
    assert exportable.exportable
    assert branch.pts_middle_encoder is exportable
