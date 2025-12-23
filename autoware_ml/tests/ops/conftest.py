# Copyright 2025 TIER IV, Inc.
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

"""Shared fixtures for CUDA operations tests."""

from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch


@dataclass
class BEVGridConfig:
    """Configuration for BEV grid dimensions.

    Attributes:
        B: Batch size.
        D: Depth dimension.
        H: Height dimension.
        W: Width dimension.
        C: Number of feature channels.
    """

    B: int = 1
    D: int = 4
    H: int = 4
    W: int = 4
    C: int = 8


@pytest.fixture
def bev_config() -> BEVGridConfig:
    """Default BEV grid configuration for tests."""
    return BEVGridConfig()


@pytest.fixture
def cuda_device() -> torch.device:
    """CUDA device fixture, skips test if CUDA unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def create_single_point_input(
    config: BEVGridConfig,
    h_idx: int,
    w_idx: int,
    d_idx: int,
    b_idx: int,
    feature_value: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input with a single point at specified grid location.

    Args:
        config: BEV grid configuration.
        h_idx: Height index in grid.
        w_idx: Width index in grid.
        d_idx: Depth index in grid.
        b_idx: Batch index.
        feature_value: Value to fill all feature channels with.
        device: Device to create tensors on.

    Returns:
        Tuple of (feats, coords, ranks) tensors.
    """
    feats = torch.full((1, config.C), feature_value, dtype=torch.float32, device=device)
    coords = torch.tensor([[h_idx, w_idx, d_idx, b_idx]], dtype=torch.int32, device=device)
    ranks = torch.tensor([0], dtype=torch.int64, device=device)
    return feats, coords, ranks


def create_multi_point_input(
    config: BEVGridConfig,
    coords_list: List[Tuple[int, int, int, int]],
    feature_values: List[float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input with multiple points at specified locations.

    Points with the same coordinates should be adjacent and have the same rank
    for proper grouping in the BEV pooling operation.

    Args:
        config: BEV grid configuration.
        coords_list: List of (h_idx, w_idx, d_idx, b_idx) tuples.
        feature_values: Feature value for each point (fills all channels).
        device: Device to create tensors on.

    Returns:
        Tuple of (feats, coords, ranks) tensors sorted by rank.
    """
    n_points = len(coords_list)
    assert len(feature_values) == n_points

    feats = torch.tensor(
        [[v] * config.C for v in feature_values], dtype=torch.float32, device=device
    )
    coords = torch.tensor(coords_list, dtype=torch.int32, device=device)

    # Compute ranks: points at same cell get same rank
    # Rank = b * D*H*W + d * H*W + h * W + w
    ranks = (
        coords[:, 3] * config.D * config.H * config.W
        + coords[:, 2] * config.H * config.W
        + coords[:, 0] * config.W
        + coords[:, 1]
    ).to(torch.int64)

    # Sort by rank (required for interval-based pooling)
    sort_indices = torch.argsort(ranks)
    feats = feats[sort_indices]
    coords = coords[sort_indices]
    ranks = ranks[sort_indices]

    return feats, coords, ranks


def create_uniform_grid_input(
    config: BEVGridConfig,
    feature_value: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input with one point per grid cell, all with same feature value.

    Args:
        config: BEV grid configuration.
        feature_value: Value to fill all feature channels with.
        device: Device to create tensors on.

    Returns:
        Tuple of (feats, coords, ranks) tensors.
    """
    coords_list = []
    for b in range(config.B):
        for d in range(config.D):
            for h in range(config.H):
                for w in range(config.W):
                    coords_list.append((h, w, d, b))

    n_points = len(coords_list)
    feature_values = [feature_value] * n_points
    return create_multi_point_input(config, coords_list, feature_values, device)
