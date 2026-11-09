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
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.encoders.sparse.ptv3_middle_encoder import PTv3MiddleEncoder
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
)
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.point_cloud.structures import Point

requires_cuda_spconv = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_middle_encoder() -> PTv3MiddleEncoder:
    """Build the two stage test middle encoder."""
    return PTv3MiddleEncoder(
        encoder=build_ptv3_encoder(), grid_size=1.0, point_cloud_range=POINT_CLOUD_RANGE
    )


def _build_neck(keep_deepest: bool = False) -> PTv3Det3DFeaturePyramid:
    """Build the pyramid fusing the two test stages."""
    return PTv3Det3DFeaturePyramid(
        stage_channels=(8, 16), out_channels=32, keep_deepest=keep_deepest
    )


class _ScalingBackbone(nn.Module):
    """Scale the deepest stage features by one learnable factor."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, point: Point) -> Point:
        point.feat = point.feat * self.scale
        return point


def _build_branch(
    keep_deepest: bool = False, pts_backbone: nn.Module | None = None
) -> Sparse4DLidar:
    """Build a small branch over the two stage test middle encoder, identity backbone by default."""
    return Sparse4DLidar(
        pts_voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        pts_middle_encoder=_build_middle_encoder(),
        pts_backbone=nn.Identity() if pts_backbone is None else pts_backbone,
        pts_neck=_build_neck(keep_deepest=keep_deepest),
    )


@requires_cuda_spconv
def test_forward_returns_one_stage_per_level_finest_first() -> None:
    device = torch.device("cuda")
    branch = _build_branch(keep_deepest=True).to(device)
    voxels_data = build_inputs(device=device).voxels_data

    stages = branch(
        voxels=voxels_data.voxels,
        coords=voxels_data.batch_xyz_coords(),
        num_points=voxels_data.num_points,
        batch_size=1,
    )

    assert len(stages) == 2
    fused_feat, fused_grid_coord, fused_offset = stages[0]
    deepest_feat, deepest_grid_coord, deepest_offset = stages[1]
    assert fused_feat.shape[1] == 32 and deepest_feat.shape[1] == 32
    assert fused_feat.shape[0] == fused_grid_coord.shape[0] == voxels_data.voxels.shape[0]
    assert deepest_feat.shape[0] == deepest_grid_coord.shape[0] <= fused_feat.shape[0]
    assert fused_offset.tolist() == [fused_feat.shape[0]]
    assert deepest_offset.tolist() == [deepest_feat.shape[0]]


@requires_cuda_spconv
def test_backbone_refines_the_encoder_stages_before_the_neck() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    plain = _build_branch().to(device).eval()
    torch.manual_seed(0)
    refined = _build_branch(pts_backbone=_ScalingBackbone()).to(device).eval()
    voxels_data = build_inputs(device=device).voxels_data

    with torch.no_grad():
        outputs = [
            branch(
                voxels=voxels_data.voxels,
                coords=voxels_data.batch_xyz_coords(),
                num_points=voxels_data.num_points,
                batch_size=1,
            )[0]
            for branch in (plain, refined)
        ]

    (plain_feat, _, _), (refined_feat, _, _) = outputs
    assert plain_feat.shape == refined_feat.shape
    assert not torch.allclose(plain_feat, refined_feat)
