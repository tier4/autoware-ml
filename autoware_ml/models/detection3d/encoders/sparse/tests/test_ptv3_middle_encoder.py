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

"""Unit tests for the PTv3 middle encoder."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.encoders.sparse.ptv3_middle_encoder import PTv3MiddleEncoder
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import collect_encoder_stage_points
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

requires_cuda_spconv = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_middle_encoder(freeze_encoder: bool = False) -> PTv3MiddleEncoder:
    """Build the two stage test middle encoder over the 8 m test range."""
    return PTv3MiddleEncoder(
        encoder=build_ptv3_encoder(),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
        freeze_encoder=freeze_encoder,
    )


def test_frozen_encoder_stays_in_eval_mode_without_gradients() -> None:
    middle_encoder = _build_middle_encoder(freeze_encoder=True)

    middle_encoder.train()

    assert middle_encoder.training
    assert not middle_encoder.encoder.training
    assert all(not parameter.requires_grad for parameter in middle_encoder.encoder.parameters())


def test_optimizer_groups_split_the_attention_blocks_from_the_rest() -> None:
    middle_encoder = _build_middle_encoder()

    groups = middle_encoder.build_optimizer_groups()

    assert set(groups) == {"encoder_default", "encoder_block"}
    grouped = sum(len(parameters) for parameters in groups.values())
    assert grouped == sum(1 for parameter in middle_encoder.parameters() if parameter.requires_grad)
    assert len(groups["encoder_block"]) > 0


def test_encoder_inputs_read_the_position_and_the_batch_from_the_voxels() -> None:
    voxel_features = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [1, 7, 8, 9], [1, 1, 1, 1]])

    inputs = PTv3MiddleEncoder.build_encoder_inputs(voxel_features, coords)

    assert torch.equal(inputs["coord"], voxel_features[:, :3])
    assert torch.equal(inputs["feat"], voxel_features)
    assert torch.equal(inputs["grid_coord"], coords[:, 1:])
    assert inputs["offset"].tolist() == [2, 4]


@requires_cuda_spconv
def test_forward_returns_the_deepest_stage_with_the_pooling_chain() -> None:
    device = torch.device("cuda")
    middle_encoder = _build_middle_encoder().to(device)
    voxel_encoder = SweepSplitVoxelFeatureEncoder().to(device)
    voxels_data = build_inputs(device=device).voxels_data

    point = middle_encoder(
        voxel_features=voxel_encoder(voxels_data.voxels, voxels_data.num_points),
        coords=voxels_data.batch_xyz_coords(),
    )

    stages = collect_encoder_stage_points(point)
    assert len(stages) == 2
    assert stages[0].feat.shape == (voxels_data.voxels.shape[0], 8)
    assert point.feat.shape[1] == 16


@requires_cuda_spconv
def test_export_context_runs_the_encoder_over_every_stage() -> None:
    device = torch.device("cuda")
    middle_encoder = _build_middle_encoder().to(device).eval()
    voxel_encoder = SweepSplitVoxelFeatureEncoder().to(device)

    context = middle_encoder.build_export_context(voxel_encoder, build_inputs(device=device))

    assert context.stage_count == 2
    assert context.strides == (2,)
    assert context.encoder_input_names[:3] == ["voxels", "num_points_per_voxel", "grid_coord"]
    assert context.stage_feats[0].shape[1] == 8
