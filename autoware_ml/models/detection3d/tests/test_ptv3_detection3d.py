"""Unit tests for PTv3-based detection models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.models.detection3d.ptv3 import (
    PTv3Det3DFeatureFusion,
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


def test_feature_pyramid_fuses_top_down_like_the_single_fusion() -> None:
    feats, clusters, grid_coords = _point_chain()
    point = rebuild_point_chain(feats, clusters, grid_coords)
    pyramid = PTv3Det3DFeaturePyramid(stage_channels=(4, 8, 16), out_channels=6).eval()
    with torch.no_grad():
        levels = pyramid(point)

    assert pyramid.num_levels == 2
    assert [level[0].shape for level in levels] == [(12, 6), (6, 6)]
    assert torch.equal(levels[0][1], grid_coords[0])
    # Reading the chain leaves it intact for the segmentation decoder
    assert point["pooling_parent"]["pooling_parent"]["feat"] is feats[0]

    # One level is the fusion of the deepest stage into its parent
    fusion = PTv3Det3DFeatureFusion(in_channels=16, skip_channels=8, out_channels=6).eval()
    single = PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=6).eval()
    single.proj_coarse[0].load_state_dict(fusion.proj.state_dict())
    single.proj_skip[0].load_state_dict(fusion.proj_skip.state_dict())
    coarse = rebuild_point_chain(feats[1:], clusters[1:], grid_coords[1:])
    with torch.no_grad():
        assert torch.allclose(single(coarse)[0][0], fusion(coarse)[0])


def test_feature_pyramid_keeps_the_projected_deepest_stage_as_its_coarsest_level() -> None:
    """
    Input: a three stage chain with the voxel coordinates of every stage, read by one pyramid
    that keeps the deepest stage and one that does not, sharing their weights.
    Expected: keeping the deepest stage adds no weights and leaves the fused levels untouched,
    it only returns the projection that feeds the first fusion as the coarsest level.
    Check: same state dict keys, equal fused levels, and a last level equal to the deepest
    projection on the deepest voxel coordinates.
    """
    feats, clusters, grid_coords = _point_chain()
    deepest_grid_coord = torch.randint(0, 2, (3, 3), generator=torch.Generator().manual_seed(1))
    fused_only = PTv3Det3DFeaturePyramid(stage_channels=(4, 8, 16), out_channels=6).eval()
    keeping = PTv3Det3DFeaturePyramid(
        stage_channels=(4, 8, 16), out_channels=6, keep_deepest=True
    ).eval()
    keeping.load_state_dict(fused_only.state_dict())

    with torch.no_grad():
        fused_levels = fused_only(rebuild_point_chain(feats, clusters, grid_coords))
        levels = keeping(rebuild_point_chain(feats, clusters, [*grid_coords, deepest_grid_coord]))
        deepest = keeping.proj_coarse[0](feats[-1])

    assert keeping.state_dict().keys() == fused_only.state_dict().keys()
    assert (keeping.num_stages, keeping.num_levels) == (3, 3)
    for kept, fused in zip(levels[:-1], fused_levels, strict=True):
        assert torch.equal(kept[0], fused[0])
    assert torch.equal(levels[-1][0], deepest)
    assert torch.equal(levels[-1][1], deepest_grid_coord)


def test_point_chain_rejects_voxel_coordinates_of_neither_layout() -> None:
    """
    Input: a three stage chain given the voxel coordinates of one stage only.
    Expected: the coordinates cover every stage but the deepest, or every stage, nothing else.
    Check: a ValueError naming both accepted counts is raised.
    """
    feats, clusters, grid_coords = _point_chain()

    with pytest.raises(ValueError, match="2 or 3 stages"):
        rebuild_point_chain(feats, clusters, grid_coords[:1])
