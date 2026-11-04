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

"""Sparse deformable aggregation of multi-level voxel features at keypoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes
from torch.utils.checkpoint import checkpoint

# Row of the zero feature that backs the empty cells, a constant so the lookup exports to ONNX
EMPTY_ROW = 0


@dataclass(frozen=True)
class SparseLevel:
    """Features of one sparse encoder stage with the voxel lookup of its dense grid.

    Attributes:
        feat: Stage features ``(V + 1, C)``, the first row is zeros and backs the empty cells.
        index_grid: Row of every dense grid cell of every sample ``(B * cells,)``, the cell
            ``(x, y, z)`` of sample ``b`` at ``((b * Z + z) * Y + y) * X + x``.
        grid_dims: Cells of the dense grid along ``(x, y, z)``.
        grid_origin: Metric origin of the grid.
        voxel_size: Metric size of a stage voxel.
    """

    feat: torch.Tensor
    index_grid: torch.Tensor
    grid_dims: tuple[int, int, int]
    grid_origin: tuple[float, float, float]
    voxel_size: float

    def sample(self, key_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the stage trilinearly at metric keypoints, empty cells read zeros.

        Args:
            key_points: Metric keypoints ``(B, N, P, 3)``.

        Returns:
            Sampled features ``(B, N, P, C)`` and the interpolation weight on occupied voxels
            ``(B, N, P)``.
        """
        return _sample_level(
            self.feat,
            self.index_grid,
            key_points,
            self.grid_dims,
            self.grid_origin,
            self.voxel_size,
        )


def _sample_level(
    feat: torch.Tensor,
    index_grid: torch.Tensor,
    key_points: torch.Tensor,
    grid_dims: tuple[int, int, int],
    grid_origin: tuple[float, float, float],
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate the stage features of the 8 cells around every keypoint, see :class:`SparseLevel`."""
    size_x, size_y, size_z = grid_dims
    num_cells = size_x * size_y * size_z
    batch_size = key_points.shape[0]
    continuous = (key_points - key_points.new_tensor(grid_origin)) / voxel_size - 0.5
    base_cell = continuous.floor()
    fraction = continuous - base_cell
    corners = torch.tensor(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=feat.device
    )
    cells = base_cell.long()[..., None, :] + corners
    corner_weight = torch.where(
        corners.bool(), fraction[..., None, :], 1.0 - fraction[..., None, :]
    ).prod(dim=-1)
    in_bounds = ((cells >= 0) & (cells < torch.tensor(grid_dims, device=feat.device))).all(dim=-1)
    batch_offsets = torch.arange(batch_size, device=feat.device).view(-1, 1, 1, 1) * num_cells
    flat = ((cells[..., 2] * size_y + cells[..., 1]) * size_x + cells[..., 0]) + batch_offsets
    rows = torch.where(in_bounds, index_grid[flat.clamp(0, batch_size * num_cells - 1)], EMPTY_ROW)
    # The empty row gathers no gradient, so the many empty corners cost nothing in backward
    gathered = F.embedding(rows, feat, padding_idx=EMPTY_ROW)
    occupancy = (corner_weight * (rows != EMPTY_ROW)).sum(dim=-1)
    return (gathered * corner_weight[..., None]).sum(dim=-2), occupancy


def _sample_stage(
    level: SparseLevel, key_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one stage, recomputing the gather in the backward pass to keep one tensor per stage."""
    if torch.is_grad_enabled():
        return checkpoint(
            _sample_level,
            level.feat,
            level.index_grid,
            key_points,
            level.grid_dims,
            level.grid_origin,
            level.voxel_size,
            use_reentrant=False,
        )
    return level.sample(key_points)


def _aggregate(
    key_points: torch.Tensor,
    weights: torch.Tensor,
    levels: Sequence[SparseLevel],
    min_occupancy: float,
) -> torch.Tensor:
    """Weighted sum of the samples of every level, divided by the weight on occupied voxels."""
    batch_size, num_anchor, num_samples, num_groups = weights.shape
    channels = levels[0].feat.shape[1]
    samples = [_sample_stage(level, key_points) for level in levels]
    sampled = torch.cat([features for features, _ in samples], dim=2).reshape(
        batch_size, num_anchor, num_samples, num_groups, channels // num_groups
    )
    occupancy = torch.cat([occupied for _, occupied in samples], dim=2)
    occupied_weight = (weights * occupancy[..., None]).sum(dim=2).clamp_min(min_occupancy)
    return ((sampled * weights[..., None]).sum(dim=2) / occupied_weight[..., None]).reshape(
        batch_size, num_anchor, channels
    )


class _SparseDeformableAggregation(Function):
    """Bridge the multi-level aggregation into eager execution and one ONNX node."""

    @staticmethod
    def symbolic(
        g,
        key_points: torch.Value,
        weights: torch.Value,
        grid_dims: tuple[int, ...],
        voxel_sizes: tuple[float, ...],
        grid_origin: tuple[float, float, float],
        min_occupancy: float,
        *level_tensors: torch.Value,
    ):
        """Emit ``autoware::SparseDeformableAggregation`` with the stage features then the grids."""
        output = g.op(
            "autoware::SparseDeformableAggregation",
            key_points,
            weights,
            *level_tensors,
            grid_dims_i=list(grid_dims),
            voxel_sizes_f=list(voxel_sizes),
            grid_origin_f=list(grid_origin),
            min_occupancy_f=min_occupancy,
            outputs=1,
        )
        key_shape = _get_tensor_sizes(key_points)
        feat_shape = _get_tensor_sizes(level_tensors[0])
        output.setType(level_tensors[0].type().with_sizes([*key_shape[:2], feat_shape[1]]))
        return output

    @staticmethod
    def forward(
        ctx,
        key_points: torch.Tensor,
        weights: torch.Tensor,
        grid_dims: tuple[int, ...],
        voxel_sizes: tuple[float, ...],
        grid_origin: tuple[float, float, float],
        min_occupancy: float,
        *level_tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Run the aggregation in eager mode, it traces the export shapes."""
        num_levels = len(voxel_sizes)
        levels = [
            SparseLevel(
                feat=level_tensors[level],
                index_grid=level_tensors[num_levels + level],
                grid_dims=tuple(grid_dims[3 * level : 3 * level + 3]),
                grid_origin=grid_origin,
                voxel_size=voxel_sizes[level],
            )
            for level in range(num_levels)
        ]
        return _aggregate(key_points, weights, levels, min_occupancy)


def sparse_deformable_aggregation(
    key_points: torch.Tensor,
    weights: torch.Tensor,
    levels: Sequence[SparseLevel],
    min_occupancy: float,
) -> torch.Tensor:
    """Aggregate multi-level voxel features at the keypoints of every anchor.

    Every keypoint interpolates trilinearly over the 8 cells around it on every level. The samples
    are summed with their weights per feature group and divided by the weight that landed on
    occupied voxels, floored at ``min_occupancy``. During ONNX export the whole aggregation is one
    ``autoware::SparseDeformableAggregation`` node, served by the TensorRT plugin of the same name.

    Args:
        key_points: Metric keypoints ``(B, N, P, 3)``.
        weights: Softmaxed sampling weights ``(B, N, L * P, G)``, level major.
        levels: Sampled stages, finest first.
        min_occupancy: Floor of the occupied weight.

    Returns:
        Aggregated features ``(B, N, C)``.

    Raises:
        ValueError: If an export gets more than one sample or levels with different origins.
    """
    if not torch.onnx.is_in_onnx_export():
        return _aggregate(key_points, weights, levels, min_occupancy)
    if key_points.shape[0] != 1:
        raise ValueError(f"The aggregation exports one sample, got {key_points.shape[0]}.")
    origins = {level.grid_origin for level in levels}
    if len(origins) != 1:
        raise ValueError(f"The aggregated levels must share one grid origin, got {origins}.")
    return _SparseDeformableAggregation.apply(
        key_points,
        weights,
        tuple(dim for level in levels for dim in level.grid_dims),
        tuple(level.voxel_size for level in levels),
        levels[0].grid_origin,
        min_occupancy,
        *(level.feat for level in levels),
        *(level.index_grid for level in levels),
    )
