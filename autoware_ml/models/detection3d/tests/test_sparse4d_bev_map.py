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

"""Tests for the Sparse4D head sampling a BEV map placed on the voxel grid."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)


def test_bev_sampling_takes_exactly_one_bev_source(tmp_path: Path) -> None:
    anchor_file = make_anchor_file(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        build_sparse4d_head(anchor_file, lidar_sampling="bev", bev_channels=32)
    with pytest.raises(ValueError, match="exactly one"):
        build_sparse4d_head(anchor_file, lidar_sampling="sparse", bev_cell_size=1.0)


def test_the_map_becomes_one_level_on_the_voxel_grid(tmp_path: Path) -> None:
    head = build_sparse4d_head(
        make_anchor_file(tmp_path), lidar_sampling="bev", bev_cell_size=2.0, bev_channels=32
    )
    bev_map = torch.randn(2, 32, 4, 4)

    levels = head.bev_levels(bev_map)

    assert len(levels) == 1
    assert levels[0].feat is bev_map
    assert levels[0].grid_origin == (POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1])
    assert levels[0].cell_size == 2.0
    assert head.levels(bev_map) == []
    assert all(block.bev_aggregation.num_levels == 1 for block in head.deformable)


def test_a_map_placing_head_refuses_the_sparse_stages(tmp_path: Path) -> None:
    head = build_sparse4d_head(
        make_anchor_file(tmp_path), lidar_sampling="bev", bev_cell_size=1.0, bev_channels=32
    )
    stage = (torch.zeros(3, 32), torch.zeros(3, 3, dtype=torch.long), torch.tensor([3]))

    with pytest.raises(ValueError, match="BEV map"):
        head.bev_levels([stage])
