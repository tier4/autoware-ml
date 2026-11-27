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

"""Unit tests of the fused PTv3 stages the SparseDrive detection head samples."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.ptv3 import rebuild_point_chain
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import build_point_chain
from autoware_ml.models.sparsedrive.lidar_stages import PTv3Det3DFeaturePyramid


def test_feature_pyramid_fuses_top_down_and_keeps_the_pooling_chain() -> None:
    """
    Input: a three stage chain read by a pyramid.
    Expected: every fused stage is its skip projection plus the fused coarser stage gathered
    through the pooling cluster, the chain stays intact for the segmentation decoder.
    """
    feats, clusters, grid_coords = build_point_chain()
    point = rebuild_point_chain(feats, clusters, grid_coords)
    pyramid = PTv3Det3DFeaturePyramid(stage_channels=(4, 8, 16), out_channels=6).eval()
    with torch.no_grad():
        levels = pyramid(point)
        middle = pyramid.proj_skip[1](feats[1]) + pyramid.proj_coarse[0](feats[2])[clusters[1]]
        finest = pyramid.proj_skip[0](feats[0]) + pyramid.proj_coarse[1](middle)[clusters[0]]

    assert pyramid.num_levels == 2
    assert [level.feat.shape for level in levels] == [(12, 6), (6, 6)]
    torch.testing.assert_close(levels[1].feat, middle)
    torch.testing.assert_close(levels[0].feat, finest)
    assert torch.equal(levels[0].grid_coord, grid_coords[0])
    assert torch.equal(levels[1].grid_coord, grid_coords[1])
    # Reading the chain leaves it intact for the segmentation decoder
    assert point["pooling_parent"]["pooling_parent"]["feat"] is feats[0]


def test_feature_pyramid_needs_two_stages() -> None:
    with pytest.raises(ValueError, match="at least two stages"):
        PTv3Det3DFeaturePyramid(stage_channels=(4,), out_channels=6)
