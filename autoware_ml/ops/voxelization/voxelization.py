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

"""Vectorized hard voxelization for PointPillars-style detectors.

Replaces the Python-level point loop with GPU-native PyTorch scatter operations.
Runs entirely on the input tensor's device.
"""

from __future__ import annotations

import torch


def hard_voxelize(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_num_points: int,
    max_voxels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize a point cloud into fixed-size pillars.

    Points are assigned to voxel cells defined by ``voxel_size`` and
    ``point_cloud_range``. Each voxel retains at most ``max_num_points``
    points. At most ``max_voxels`` occupied voxels are kept. When more voxels
    are occupied than ``max_voxels``, the retained set is the first
    ``max_voxels`` in lexicographic ZYX key order.

    The implementation is fully vectorized and executes on the same device as
    ``points``. No Python-level loops or CPU–GPU transfers are required.

    Args:
        points: Input point cloud with shape ``(N, C)``. The first three
            columns are XYZ coordinates in meters.
        voxel_size: Per-axis voxel size ``[dx, dy, dz]`` in meters, shape ``(3,)``.
        point_cloud_range: Spatial range ``[x_min, y_min, z_min, x_max, y_max, z_max]``
            in meters, shape ``(6,)``. Points outside this range are discarded.
        max_num_points: Maximum number of points retained per voxel.
        max_voxels: Maximum number of occupied voxels returned.

    Returns:
        Tuple of three tensors:

        - **voxels** ``(M, max_num_points, C)`` - padded point features.
          Empty slots are zero-filled.
        - **coords** ``(M, 3)`` - integer voxel coordinates in ZYX order,
          ``dtype=torch.int32``.
        - **num_points** ``(M,)`` - number of valid points per voxel,
          capped at ``max_num_points``, ``dtype=torch.int32``.

        where ``M = min(num_occupied_voxels, max_voxels)``.
    """
    device = points.device
    C = points.shape[1]

    lower = point_cloud_range[:3]
    upper = point_cloud_range[3:]

    # Integer grid coordinates (XYZ) and grid extents
    # Use round() to match mmcv's grid_size convention.
    grid_size = torch.round((upper - lower) / voxel_size).to(torch.int64)  # [Nx, Ny, Nz]
    grid_coords = torch.floor((points[:, :3] - lower) / voxel_size).to(torch.int64)

    # Filter points inside the grid
    # Bounds-check the integer coordinates rather than the metric positions:
    # float rounding at the upper range boundary can otherwise yield
    # coords == grid_size, which corrupts downstream scatter indices.
    valid = ((grid_coords >= 0) & (grid_coords < grid_size)).all(dim=1)
    points = points[valid]
    grid_coords = grid_coords[valid]

    if points.shape[0] == 0:
        return (
            points.new_zeros((0, max_num_points, C)),
            torch.zeros((0, 3), device=device, dtype=torch.int32),
            torch.zeros((0,), device=device, dtype=torch.int32),
        )

    # Flat voxel key in ZYX order (z varies slowest)
    # key = z * Ny * Nx  +  y * Nx  +  x
    keys = (
        grid_coords[:, 2] * grid_size[1] * grid_size[0]
        + grid_coords[:, 1] * grid_size[0]
        + grid_coords[:, 0]
    )

    # Sort points by voxel key
    sorted_keys, sort_idx = torch.sort(keys, stable=True)
    sorted_points = points[sort_idx]
    # ZYX coordinate order to match mmcv output convention
    sorted_zyx = grid_coords[sort_idx][:, [2, 1, 0]].to(torch.int32)

    # Find unique voxels; get per-point voxel ID and per-voxel counts
    _, voxel_id, counts = torch.unique_consecutive(
        sorted_keys, return_inverse=True, return_counts=True
    )
    M_total = counts.shape[0]
    M = min(M_total, max_voxels)

    # Unique voxel coordinates (ZYX) - first point of each unique voxel
    voxel_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]
    )
    unique_coords = sorted_zyx[voxel_starts[:M]]

    # Assign within-voxel slot to every point
    point_idx = torch.arange(sorted_keys.shape[0], device=device)
    # slot[i] = position of sorted point i within its voxel group
    slot = point_idx - voxel_starts[voxel_id]

    # Keep points that fall within max_voxels and max_num_points limits
    keep = (voxel_id < M) & (slot < max_num_points)
    valid_voxel_ids = voxel_id[keep]  # int64
    valid_slots = slot[keep]  # int64
    valid_points = sorted_points[keep]

    # Fill voxels tensor via scatter
    voxels = points.new_zeros((M, max_num_points, C))
    flat_idx = valid_voxel_ids * max_num_points + valid_slots
    voxels.view(-1, C).scatter_(
        0,
        flat_idx.unsqueeze(1).expand(-1, C),
        valid_points,
    )

    # Count valid points per voxel (capped at max_num_points)
    num_points = torch.zeros(M, dtype=torch.int32, device=device)
    num_points.scatter_add_(
        0,
        valid_voxel_ids.to(torch.int32),
        torch.ones(valid_voxel_ids.shape[0], dtype=torch.int32, device=device),
    )

    return voxels, unique_coords, num_points
