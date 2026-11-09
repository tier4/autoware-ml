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

"""Unit tests for the BEV map pyramid."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.necks.bev_map_pyramid import BEVMapPyramid

POINT_CLOUD_RANGE = [-8.0, -4.0, -2.0, 8.0, 4.0, 2.0]


def _build_pyramid(num_downsampled: int = 1) -> BEVMapPyramid:
    return BEVMapPyramid(
        point_cloud_range=POINT_CLOUD_RANGE,
        voxel_size=[0.5, 0.5, 4.0],
        out_size_factor=2,
        in_channels=6,
        channels=4,
        num_downsampled=num_downsampled,
    )


def test_cells_must_be_square() -> None:
    with pytest.raises(ValueError, match="square"):
        BEVMapPyramid(
            point_cloud_range=POINT_CLOUD_RANGE,
            voxel_size=[0.5, 0.25, 4.0],
            out_size_factor=2,
            in_channels=6,
            channels=4,
        )


def test_levels_halve_the_projected_map_on_the_metric_grid() -> None:
    pyramid = _build_pyramid(num_downsampled=1).eval()
    bev_map = torch.randn(2, 6, 8, 16)

    levels = pyramid(bev_map)

    assert pyramid.num_levels == 2
    assert [level.feat.shape for level in levels] == [(2, 4, 8, 16), (2, 4, 4, 8)]
    assert [level.cell_size for level in levels] == [1.0, 2.0]
    assert all(level.grid_origin == (-8.0, -4.0) for level in levels)


def test_without_downsampling_the_projected_map_is_the_only_level() -> None:
    pyramid = _build_pyramid(num_downsampled=0).eval()

    levels = pyramid(torch.randn(1, 6, 8, 16))

    assert pyramid.num_levels == 1
    assert len(levels) == 1
    assert levels[0].feat.shape == (1, 4, 8, 16)
