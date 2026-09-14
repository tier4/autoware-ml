"""Unit tests for the temporal-query helpers shared by streaming detectors."""

from __future__ import annotations

import math

import torch

from autoware_ml.models.detection3d.task_modules.streaming import (
    inverse_sigmoid,
    memory_refresh,
    pos2posemb1d,
    pos2posemb3d,
    topk_gather,
    transform_reference_points,
)


def test_pos2posemb3d_concatenates_y_x_z_in_reference_order() -> None:
    # The (y, x, z) concatenation order is what the pretrained StreamPETR
    # weights expect; swapping it silently degrades a loaded checkpoint.
    positions = torch.tensor([[[0.1, 0.7, 0.4]]])

    embedding = pos2posemb3d(positions, num_pos_feats=4)

    assert embedding.shape == (1, 1, 12)
    expected_y = pos2posemb1d(positions[..., 1:2], num_pos_feats=4)
    expected_x = pos2posemb1d(positions[..., 0:1], num_pos_feats=4)
    expected_z = pos2posemb1d(positions[..., 2:3], num_pos_feats=4)
    torch.testing.assert_close(embedding[..., 0:4], expected_y)
    torch.testing.assert_close(embedding[..., 4:8], expected_x)
    torch.testing.assert_close(embedding[..., 8:12], expected_z)


def test_pos2posemb1d_is_scaled_by_two_pi_and_interleaves_sin_cos() -> None:
    embedding = pos2posemb1d(torch.tensor([[[0.25]]]), num_pos_feats=2)

    angle = 0.25 * 2.0 * math.pi
    torch.testing.assert_close(embedding[0, 0, 0], torch.tensor(math.sin(angle)))
    torch.testing.assert_close(embedding[0, 0, 1], torch.tensor(math.cos(angle)))


def test_memory_refresh_zeroes_entries_of_discontinued_streams() -> None:
    memory = torch.ones(2, 3, 4)
    prev_exists = torch.tensor([1.0, 0.0])

    refreshed = memory_refresh(memory, prev_exists)

    assert refreshed[0].eq(1.0).all()
    assert refreshed[1].eq(0.0).all()


def test_topk_gather_selects_the_same_entries_across_feature_dims() -> None:
    features = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    indices = torch.tensor([[3, 0], [1, 2]])

    gathered = topk_gather(features, indices)

    assert gathered.shape == (2, 2, 3)
    torch.testing.assert_close(gathered[0], features[0][[3, 0]])
    torch.testing.assert_close(gathered[1], features[1][[1, 2]])
    # A missing index tensor is a pass-through.
    assert topk_gather(features, None) is features


def test_transform_reference_points_applies_the_pose_per_sample() -> None:
    points = torch.tensor([[[1.0, 2.0, 3.0]]])
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, 3] = torch.tensor([10.0, -5.0, 0.5])

    moved = transform_reference_points(points, pose)

    torch.testing.assert_close(moved, torch.tensor([[[11.0, -3.0, 3.5]]]))


def test_inverse_sigmoid_inverts_sigmoid_and_clamps_extremes() -> None:
    values = torch.tensor([0.2, 0.5, 0.9])

    torch.testing.assert_close(inverse_sigmoid(values).sigmoid(), values, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(inverse_sigmoid(torch.tensor([0.0, 1.0, -1.0, 2.0]))).all()
