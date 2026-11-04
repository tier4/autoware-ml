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

"""Tests for the BEV deformable aggregation operator."""

from __future__ import annotations

import io

import onnx
import torch
import torch.nn as nn

from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import (
    BEVLevel,
    bev_deformable_aggregation,
)


def _two_by_two() -> torch.Tensor:
    """Map of one sample, two channels, rows y and columns x."""
    return torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]])


def test_bev_level_reads_cells_at_their_center_ignores_height_and_reads_zeros_outside() -> None:
    level = BEVLevel(_two_by_two(), grid_origin=(0.0, 0.0), cell_size=1.0)
    # Center of cell (x 0, y 0), of cell (x 1, y 0) high above, between them and off the map
    key_points = torch.tensor(
        [[[[0.5, 0.5, 0.0], [1.5, 0.5, 9.0], [1.0, 0.5, 0.0], [-5.0, 0.0, 0.0]]]]
    )

    sampled = level.sample(key_points)

    expected = torch.tensor([[1.0, 10.0], [2.0, 20.0], [1.5, 15.0], [0.0, 0.0]])
    assert sampled.shape == (1, 1, 4, 2)
    assert torch.allclose(sampled[0, 0], expected)


def test_aggregation_weights_every_level_and_keypoint_per_group() -> None:
    fine = BEVLevel(_two_by_two(), grid_origin=(0.0, 0.0), cell_size=1.0)
    coarse = BEVLevel(torch.full((1, 2, 1, 1), 7.0), grid_origin=(0.0, 0.0), cell_size=2.0)
    # The first keypoint sits at the center of the coarse cell, the second of a fine one
    key_points = torch.tensor([[[[1.0, 1.0, 0.0], [1.5, 1.5, 0.0]]]])
    # Two groups of one channel: the first reads the second fine sample, the second the first
    # coarse sample
    weights = torch.zeros((1, 1, 4, 2))
    weights[0, 0, 1, 0] = 1.0
    weights[0, 0, 2, 1] = 1.0

    features = bev_deformable_aggregation(key_points, weights, [fine, coarse])

    assert torch.allclose(features, torch.tensor([[[4.0, 7.0]]]))


class _BEVAggregation(nn.Module):
    """One level aggregation at the export boundary."""

    def forward(
        self, key_points: torch.Tensor, weights: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        return bev_deformable_aggregation(
            key_points, weights, [BEVLevel(feat, grid_origin=(-4.0, -4.0), cell_size=0.5)]
        )


def test_export_samples_the_map_with_onnx_grid_sample() -> None:
    inputs = (
        torch.rand(1, 5, 3, 3),
        torch.rand(1, 5, 3, 2).softmax(dim=-2),
        torch.rand(1, 8, 16, 16),
    )
    buffer = io.BytesIO()

    torch.onnx.export(
        _BEVAggregation(),
        inputs,
        buffer,
        input_names=["key_points", "weights", "bev"],
        output_names=["features"],
        opset_version=17,
        dynamo=False,
    )

    graph = onnx.load_from_string(buffer.getvalue()).graph
    op_types = {node.op_type for node in graph.node}
    assert "GridSample" in op_types
    assert all(node.domain in ("", "ai.onnx") for node in graph.node)
