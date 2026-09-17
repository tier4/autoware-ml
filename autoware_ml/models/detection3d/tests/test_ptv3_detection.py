"""Unit tests for PTv3-based detection models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import (
    build_inputs,
    build_seg_model,
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
def test_ptv3_transhead_detection_runs_loss_and_predict() -> None:
    model = build_trans_model().to(torch.device("cuda"))
    batch = build_inputs(device=torch.device("cuda"))

    outputs = model(**model.bind_forward_inputs(batch))
    metrics = model.compute_metrics(batch, outputs)
    predictions = model.bbox_head.predict(outputs)

    assert "loss" in metrics
    assert outputs["dense_heatmap"].shape[:2] == (1, 2)
    assert outputs["query_labels"].shape == (1, 8)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Detection3DSamplePredictions)


def test_ptv3_detection_loads_encoder_from_seg_checkpoint_via_matching_weights(
    tmp_path: Path,
) -> None:
    segmentation_model = build_seg_model()
    checkpoint_path = tmp_path / "ptv3_segmentation.ckpt"
    torch.save({"state_dict": segmentation_model.state_dict()}, checkpoint_path)

    model = build_trans_model(freeze_encoder=True)
    apply_matching_weights(model, (checkpoint_path,))
    model.train()

    reference_state = segmentation_model.encoder.state_dict()
    loaded_state = model.encoder.state_dict()
    first_key = next(iter(reference_state))

    assert torch.allclose(loaded_state[first_key], reference_state[first_key])
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.encoder.training is False
