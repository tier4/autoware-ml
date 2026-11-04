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

"""Unit tests for PTv3-based detection models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.models.detection3d.ptv3 import (
    PTv3Det3DFeaturePyramid,
    rebuild_point_chain,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_seg3d_model,
    build_trans_model,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.checkpoints import apply_matching_weights


def test_ptv3_bev_projection_assume_valid_matches_guarded_path_for_valid_coords() -> None:
    guarded = build_trans_model().bev_neck.bev_projector.eval()
    unguarded = build_trans_model().bev_neck.bev_projector.eval()
    unguarded.load_state_dict(guarded.state_dict())
    unguarded.assume_valid_grid_coord = True

    expected_bev_shape = (8, 8)
    assert guarded.output_shape == expected_bev_shape
    bev_height, bev_width = expected_bev_shape

    point_features = torch.randn(6, guarded.point_proj[0].in_features)
    offset = torch.tensor([point_features.shape[0]], dtype=torch.long)
    grid_coord = torch.tensor(
        [[0, 0, 0], [1, 2, 0], [1, 2, 1], [3, 4, 0], [7, 7, 0], [5, 6, 2]],
        dtype=torch.int32,
    )
    assert torch.all((grid_coord[:, 0] >= 0) & (grid_coord[:, 0] < bev_width))
    assert torch.all((grid_coord[:, 1] >= 0) & (grid_coord[:, 1] < bev_height))

    with torch.no_grad():
        expected = guarded(point_features, grid_coord, offset)
        actual = unguarded(point_features, grid_coord, offset)

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_transhead_detection3d_runs_loss_and_predict() -> None:
    model = build_trans_model().to(torch.device("cuda"))
    batch = build_inputs(device=torch.device("cuda"))

    outputs = model(batch)
    metrics = model.compute_metrics(batch, outputs)
    predictions = model.decode_outputs(batch, outputs).detection3d_predictions
    head_outputs = outputs.detection3d().transfusion_head()

    assert "loss" in metrics
    assert head_outputs.dense_heatmap.shape[:2] == (1, 2)
    assert head_outputs.query_labels.shape == (1, 8)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Detection3DSamplePredictions)


def test_ptv3_detection3d_loads_encoder_from_seg3d_checkpoint_via_matching_weights(
    tmp_path: Path,
) -> None:
    segmentation3d_model = build_seg3d_model()
    checkpoint_path = tmp_path / "ptv3_segmentation3d.ckpt"
    torch.save({"state_dict": segmentation3d_model.state_dict()}, checkpoint_path)

    model = build_trans_model(freeze_encoder=True)
    apply_matching_weights(model, (checkpoint_path,))
    model.train()

    reference_state = segmentation3d_model.encoder.state_dict()
    loaded_state = model.encoder.state_dict()
    first_key = next(iter(reference_state))

    assert torch.allclose(loaded_state[first_key], reference_state[first_key])
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.encoder.training is False


def _point_chain() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Features of three stages, the pooling clusters and the voxel coordinates of a sample."""
    generator = torch.Generator().manual_seed(0)
    feats = [torch.randn((12, 4), generator=generator), torch.randn((6, 8), generator=generator)]
    feats.append(torch.randn((3, 16), generator=generator))
    clusters = [torch.arange(12) // 2, torch.arange(6) // 2]
    grid_coords = [torch.randint(0, 8, (12, 3), generator=generator), torch.randint(0, 4, (6, 3))]
    return feats, clusters, grid_coords


def test_feature_pyramid_fuses_top_down_and_keeps_the_pooling_chain() -> None:
    """
    Input: a three stage chain read by a pyramid.
    Expected: every fused stage is its skip projection plus the fused coarser stage gathered
    through the pooling cluster, the chain stays intact for the segmentation decoder.
    """
    feats, clusters, grid_coords = _point_chain()
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


def test_point_chain_rejects_voxel_coordinates_of_another_stage_count() -> None:
    """
    Input: a three stage chain given the voxel coordinates of one stage only.
    Expected: the clusters and coordinates cover every stage but the deepest, nothing else.
    Check: a ValueError naming the expected count is raised.
    """
    feats, clusters, grid_coords = _point_chain()

    with pytest.raises(ValueError, match="of 2 stages"):
        rebuild_point_chain(feats, clusters, grid_coords[:1])
