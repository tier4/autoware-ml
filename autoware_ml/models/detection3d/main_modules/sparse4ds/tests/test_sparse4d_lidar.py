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

from autoware_ml.models.detection3d.backbones.ptv3 import PTv3SparseBackbone
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
)
from autoware_ml.models.segmentation3d.encoders.voxel import (
    MeanVoxelFeatureEncoder,
    SweepSplitVoxelFeatureEncoder,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

requires_cuda_spconv = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_backbone() -> PTv3SparseBackbone:
    """Build the two stage test backbone."""
    return PTv3SparseBackbone(
        encoder=build_ptv3_encoder(), grid_size=1.0, point_cloud_range=POINT_CLOUD_RANGE
    )


def _build_neck(keep_deepest: bool = False) -> PTv3Det3DFeaturePyramid:
    """Build the pyramid fusing the two test stages."""
    return PTv3Det3DFeaturePyramid(
        stage_channels=(8, 16), out_channels=32, keep_deepest=keep_deepest
    )


def _build_branch(keep_deepest: bool = False) -> Sparse4DLidar:
    """Build a small branch over the two stage test backbone."""
    return Sparse4DLidar(
        pts_voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        pts_backbone=_build_backbone(),
        pts_neck=_build_neck(keep_deepest=keep_deepest),
    )


def test_backbone_must_be_a_sparse_backbone() -> None:
    with pytest.raises(TypeError, match="SparseBackbone"):
        Sparse4DLidar(
            pts_voxel_encoder=SweepSplitVoxelFeatureEncoder(),
            pts_backbone=nn.Identity(),
            pts_neck=_build_neck(),
        )


def test_neck_must_be_a_sparse_neck() -> None:
    with pytest.raises(TypeError, match="SparseNeck"):
        Sparse4DLidar(
            pts_voxel_encoder=SweepSplitVoxelFeatureEncoder(),
            pts_backbone=_build_backbone(),
            pts_neck=nn.Identity(),
        )


def test_voxel_encoder_must_describe_the_features_the_backbone_reads() -> None:
    with pytest.raises(ValueError, match="features per voxel"):
        Sparse4DLidar(
            pts_voxel_encoder=MeanVoxelFeatureEncoder(),
            pts_backbone=_build_backbone(),
            pts_neck=_build_neck(),
        )


def test_num_levels_follows_the_neck() -> None:
    assert _build_branch().num_levels == 1
    assert _build_branch(keep_deepest=True).num_levels == 2


def test_optimizer_groups_split_the_backbone_from_the_neck() -> None:
    branch = _build_branch()

    groups = branch.build_optimizer_groups()

    assert set(groups) == {"encoder_default", "encoder_block", "pts_neck"}
    grouped = sum(len(parameters) for parameters in groups.values())
    assert grouped == sum(1 for parameter in branch.parameters() if parameter.requires_grad)
    assert len(groups["pts_neck"]) == sum(1 for _ in branch.pts_neck.parameters())


def test_points_must_follow_the_layout_of_the_voxel_encoder() -> None:
    branch = _build_branch()
    voxels_data = build_inputs().voxels_data

    with pytest.raises(ValueError, match="time lag"):
        branch(
            voxels=voxels_data.voxels,
            coords=voxels_data.batch_xyz_coords(),
            num_points=voxels_data.num_points,
            time_lag_column=-1,
        )


@requires_cuda_spconv
def test_forward_returns_one_stage_per_level_finest_first() -> None:
    device = torch.device("cuda")
    branch = _build_branch(keep_deepest=True).to(device)
    batch = build_inputs(device=device)
    voxels_data = batch.voxels_data
    point_batch = batch.multi_task_gt_batch.point_cloud_gt_batch

    stages = branch(
        voxels=voxels_data.voxels,
        coords=voxels_data.batch_xyz_coords(),
        num_points=voxels_data.num_points,
        time_lag_column=point_batch.timestamp_difference_dim,
    )

    assert len(stages) == 2
    fused_feat, fused_grid_coord, fused_offset = stages[0]
    deepest_feat, deepest_grid_coord, deepest_offset = stages[1]
    assert fused_feat.shape[1] == 32 and deepest_feat.shape[1] == 32
    assert fused_feat.shape[0] == fused_grid_coord.shape[0] == voxels_data.voxels.shape[0]
    assert deepest_feat.shape[0] == deepest_grid_coord.shape[0] <= fused_feat.shape[0]
    assert fused_offset.tolist() == [fused_feat.shape[0]]
    assert deepest_offset.tolist() == [deepest_feat.shape[0]]
