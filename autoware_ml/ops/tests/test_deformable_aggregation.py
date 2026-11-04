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

"""Tests for the sparse deformable aggregation operator."""

from __future__ import annotations

import io
import itertools

import onnx
import torch
import torch.nn as nn
from onnx import helper

from autoware_ml.ops.deformable_aggregation.sparse_deformable_aggregation import (
    SparseLevel,
    sparse_deformable_aggregation,
)

GRID_ORIGIN = (-1.0, -1.0, -0.5)
MIN_OCCUPANCY = 0.05


def _random_level(
    grid_dims: tuple[int, int, int], voxel_size: float, num_voxels: int, channels: int
) -> SparseLevel:
    num_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
    index_grid = torch.zeros(num_cells, dtype=torch.long)
    index_grid[torch.randperm(num_cells)[:num_voxels]] = torch.arange(1, num_voxels + 1)
    feat = torch.cat([torch.zeros(1, channels), torch.randn(num_voxels, channels)]).double()
    return SparseLevel(feat, index_grid, grid_dims, GRID_ORIGIN, voxel_size)


def _reference(
    key_points: torch.Tensor, weights: torch.Tensor, levels: list[SparseLevel]
) -> torch.Tensor:
    """Loop over anchors, samples and corners as the TensorRT plugin does."""
    _, num_anchor, num_points, _ = key_points.shape
    num_groups = weights.shape[-1]
    channels = levels[0].feat.shape[1]
    group_channels = channels // num_groups
    output = torch.zeros(1, num_anchor, channels, dtype=torch.float64)
    for anchor in range(num_anchor):
        total = torch.zeros(channels, dtype=torch.float64)
        occupied = torch.zeros(num_groups, dtype=torch.float64)
        for level_index, level in enumerate(levels):
            size_x, size_y, size_z = level.grid_dims
            for point in range(num_points):
                weight = weights[0, anchor, level_index * num_points + point]
                origin = torch.tensor(level.grid_origin, dtype=torch.float64)
                cell = (key_points[0, anchor, point] - origin) / level.voxel_size - 0.5
                base = cell.floor()
                fraction = cell - base
                for corner in itertools.product((0, 1), repeat=3):
                    x, y, z = (int(base[axis]) + corner[axis] for axis in range(3))
                    inside = 0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z
                    row = int(level.index_grid[(z * size_y + y) * size_x + x]) if inside else 0
                    if row == 0:
                        continue
                    corner_weight = 1.0
                    for axis in range(3):
                        corner_weight *= fraction[axis] if corner[axis] else 1.0 - fraction[axis]
                    total += (
                        weight.repeat_interleave(group_channels) * corner_weight * level.feat[row]
                    )
                    occupied += weight * corner_weight
        output[0, anchor] = total / occupied.clamp_min(MIN_OCCUPANCY).repeat_interleave(
            group_channels
        )
    return output


def test_aggregation_matches_the_plugin_contract() -> None:
    torch.manual_seed(0)
    levels = [
        _random_level((6, 5, 4), voxel_size=0.5, num_voxels=40, channels=4),
        _random_level((3, 3, 2), voxel_size=1.0, num_voxels=6, channels=4),
    ]
    # Keypoints reach past the grid on every side, so the bounds check is covered
    key_points = (torch.rand(1, 7, 3, 3, dtype=torch.float64) * 4.0 - 2.0) + torch.tensor(
        [0.5, 0.25, 0.5], dtype=torch.float64
    )
    weights = torch.randn(1, 7, 2 * 3, 2, dtype=torch.float64).softmax(dim=-2)

    features = sparse_deformable_aggregation(key_points, weights, levels, MIN_OCCUPANCY)

    assert torch.allclose(features, _reference(key_points, weights, levels))


class _TwoLevelAggregation(nn.Module):
    def forward(
        self,
        key_points: torch.Tensor,
        weights: torch.Tensor,
        feat_fine: torch.Tensor,
        feat_coarse: torch.Tensor,
        grid_fine: torch.Tensor,
        grid_coarse: torch.Tensor,
    ) -> torch.Tensor:
        levels = [
            SparseLevel(feat_fine, grid_fine, (4, 4, 2), GRID_ORIGIN, 0.5),
            SparseLevel(feat_coarse, grid_coarse, (2, 2, 1), GRID_ORIGIN, 1.0),
        ]
        return sparse_deformable_aggregation(key_points, weights, levels, MIN_OCCUPANCY)


def test_export_emits_one_aggregation_node_with_the_level_attributes() -> None:
    names = ["key_points", "weights", "feat_fine", "feat_coarse", "grid_fine", "grid_coarse"]
    inputs = (
        torch.rand(1, 5, 3, 3),
        torch.rand(1, 5, 6, 2).softmax(dim=-2),
        torch.rand(4, 8),
        torch.rand(2, 8),
        torch.randint(0, 4, (32,)),
        torch.randint(0, 2, (4,)),
    )
    buffer = io.BytesIO()

    torch.onnx.export(
        _TwoLevelAggregation(),
        inputs,
        buffer,
        input_names=names,
        output_names=["features"],
        opset_version=17,
        dynamo=False,
    )

    graph = onnx.load_from_string(buffer.getvalue()).graph
    (node,) = graph.node
    attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    assert (node.domain, node.op_type) == ("autoware", "SparseDeformableAggregation")
    assert list(node.input) == names
    assert list(node.output) == ["features"]
    assert attributes["grid_dims"] == [4, 4, 2, 2, 2, 1]
    assert attributes["voxel_sizes"] == [0.5, 1.0]
    assert torch.allclose(torch.tensor(attributes["grid_origin"]), torch.tensor(GRID_ORIGIN))
    assert abs(attributes["min_occupancy"] - MIN_OCCUPANCY) < 1e-7
    dims = [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]
    assert dims == [1, 5, 8]
