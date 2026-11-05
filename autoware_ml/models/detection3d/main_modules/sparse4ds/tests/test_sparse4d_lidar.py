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


def _build_branch(freeze_encoder: bool = False, keep_deepest: bool = False) -> Sparse4DLidar:
    """Build a small branch over the two stage test encoder."""
    return Sparse4DLidar(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        det_fusion=PTv3Det3DFeaturePyramid(
            stage_channels=(8, 16), out_channels=32, keep_deepest=keep_deepest
        ),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
        freeze_encoder=freeze_encoder,
    )


def test_voxel_encoder_must_describe_the_point_features_the_encoder_reads() -> None:
    with pytest.raises(ValueError, match="point features"):
        Sparse4DLidar(
            encoder=build_ptv3_encoder(),
            voxel_encoder=MeanVoxelFeatureEncoder(),
            det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
            grid_size=1.0,
            point_cloud_range=POINT_CLOUD_RANGE,
        )


def test_num_levels_follows_the_pyramid() -> None:
    assert _build_branch().num_levels == 1
    assert _build_branch(keep_deepest=True).num_levels == 2


def test_frozen_encoder_stays_in_eval_mode_without_gradients() -> None:
    branch = _build_branch(freeze_encoder=True)

    branch.train()

    assert not branch.encoder.training
    assert branch.det_fusion.training
    assert all(not parameter.requires_grad for parameter in branch.encoder.parameters())


def test_optimizer_groups_split_the_encoder_from_the_pyramid() -> None:
    branch = _build_branch()

    groups = branch.build_optimizer_groups()

    assert set(groups) == {"encoder_default", "encoder_block", "det_fusion"}
    grouped = sum(len(parameters) for parameters in groups.values())
    assert grouped == sum(1 for parameter in branch.parameters() if parameter.requires_grad)
    assert len(groups["det_fusion"]) == sum(1 for _ in branch.det_fusion.parameters())


@requires_cuda_spconv
def test_forward_returns_one_stage_per_level_finest_first() -> None:
    device = torch.device("cuda")
    branch = _build_branch(keep_deepest=True).to(device)
    batch = build_inputs(device=device)
    voxels_data = batch.voxels_data
    point_batch = batch.multi_task_gt_batch.point_cloud_gt_batch

    stages = branch(
        voxels=voxels_data.voxels,
        num_points=voxels_data.num_points,
        voxel_coords=voxels_data.batch_xyz_coords(),
        num_dropped_voxels=voxels_data.num_dropped_voxels,
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


@requires_cuda_spconv
def test_dropped_voxels_are_refused() -> None:
    device = torch.device("cuda")
    branch = _build_branch().to(device)
    batch = build_inputs(device=device)
    voxels_data = batch.voxels_data

    with pytest.raises(RuntimeError, match="dropped"):
        branch(
            voxels=voxels_data.voxels,
            num_points=voxels_data.num_points,
            voxel_coords=voxels_data.batch_xyz_coords(),
            num_dropped_voxels=torch.tensor(3, device=device),
            time_lag_column=4,
        )


@requires_cuda_spconv
def test_export_context_runs_the_encoder_over_every_stage() -> None:
    device = torch.device("cuda")
    branch = _build_branch().to(device).eval()

    context = branch.build_export_context(build_inputs(device=device))

    assert context.stage_count == 2
    assert context.strides == (2,)
    assert context.encoder_input_names[:3] == ["voxels", "num_points_per_voxel", "grid_coord"]
