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
from typing import NamedTuple

from jaxtyping import Float32, Int32, Int64, Bool
import torch


class VoxelsData(NamedTuple):
    """
    Container for hard-voxelization results.

    Attributes:
        voxels (M, max_num_points, C): Padded point features.
            C is either (x, y, z, intensity) or (x, y, z, time_lag) if C is 4. C is
            (x, y, z, intensity, time_lag) when it's 5. Empty slots are zero-filled.
        coords (M, 3): Integer voxel coordinates in (x, y, z).
        num_points (M,): Number of valid points per voxel.
        batch_indices (M,): Batch indices for each voxel.
      where M is batch_size * maximum number of voxels.
    """

    voxels: Float32[torch.Tensor, "M max_num_points C"]
    coords: Int32[torch.Tensor, "M 3"]
    num_points: Int32[torch.Tensor, " M"]
    batch_indices: Int32[torch.Tensor, " M"]


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
        valid_voxel_ids,
        torch.ones(valid_voxel_ids.shape[0], dtype=torch.int32, device=device),
    )

    return voxels, unique_coords, num_points


def batch_hard_voxelize(
    points: Float32[torch.Tensor, "number_points number_channels"],
    points_batch_indices: Int32[torch.Tensor, " number_points"],
    voxel_size: Float32[torch.Tensor, " 3"],
    point_cloud_range: Float32[torch.Tensor, " 6"],
    max_num_points: int,
    max_voxels: int,
) -> VoxelsData:
    """Voxelize batch of point clouds into fixed-size voxels/pillars.

    Points are assigned to voxel cells defined by ``voxel_size`` and
    ``point_cloud_range``. Each voxel retains at most ``max_num_points``
    points. At most ``max_voxels`` occupied voxels are kept. When more voxels
    are occupied than ``max_voxels``, the retained set is the first
    ``max_voxels`` in lexicographic ZYX key order.

    The implementation is fully vectorized in batch and executes on the same device as
    ``points``. No Python-level loops or CPU–GPU transfers are required.

    Args:
        points (Float32[torch.Tensor, "number_points number_channels"]): Batch of input point clouds.
            The first three columns are XYZ coordinates in meters.
        points_batch_indices (Int32[torch.Tensor, "number_points"]): Batch indices for each point
            in ``points``.
        voxel_size (Float32[torch.Tensor, "3"]): Per-axis voxel size ``[dx, dy, dz]`` in meters.
        point_cloud_range (Float32[torch.Tensor, "6"]): Spatial range ``[x_min, y_min, z_min, x_max, y_max, z_max]``
            in meters. Points outside this range are discarded.
        max_num_points (int): Maximum number of points retained per voxel.
        max_voxels (int): Maximum number of occupied voxels per batch returned.

    Returns:
        VoxelsData: Named tuple containing the following fields:
            - voxels (M, max_num_points, C): Padded point voxel features.
            - coords (M, 3): Integer voxel coordinates in XYZ order.
            - num_points (M,): Number of valid points per voxel.
            - batch_indices (M,): Batch indices for each voxel.
        where M = batch_size * maximum number of voxels.
    """
    if points.shape[0] != points_batch_indices.shape[0]:
        raise ValueError(
            f"points.shape[0] ({points.shape[0]}) != points_batch_indices.shape[0] ({points_batch_indices.shape[0]})"
        )
    device = points.device
    channels = points.shape[1]

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
    points_batch_indices = points_batch_indices[valid]
    grid_coords = grid_coords[valid]

    if points.shape[0] == 0:
        return VoxelsData(
            voxels=points.new_zeros((0, max_num_points, channels)),
            coords=torch.zeros((0, 3), device=points.device, dtype=torch.int32),
            num_points=torch.zeros((0,), device=points.device, dtype=torch.int32),
            batch_indices=torch.zeros((0,), device=points.device, dtype=torch.int32),
        )

    # Flat voxel key in (Batch, Z, Y, X) order (Z varies slowest within a batch)
    # key = batch_index * Nz * Ny * Nx  +  z * Ny * Nx  +  y * Nx  +  x
    # Given tuples of (batch_idx, x, y, z):
    # (0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0),
    # the sorted keys are: 0, 1, 2, 3, 4, 5
    keys = (
        points_batch_indices.to(torch.int64) * grid_size[2] * grid_size[1] * grid_size[0]
        + grid_coords[:, 2] * grid_size[1] * grid_size[0]
        + grid_coords[:, 1] * grid_size[0]
        + grid_coords[:, 0]
    )

    # Stable (deterministic) sorting points by voxel key
    sorted_keys, sort_idx = torch.sort(keys, stable=True)
    sorted_points = points[sort_idx]
    sorted_batch_indices = points_batch_indices[sort_idx].to(torch.int32)
    sorted_grid_coords = grid_coords[sort_idx].to(torch.int32)

    # ZYX coordinate order to match mmcv output convention
    # sorted_zyx = grid_coords[sort_idx][:, [2, 1, 0]].to(torch.int32)

    # Find unique voxels; get per-point voxel ID and per-voxel counts
    _, voxel_id, counts = torch.unique_consecutive(
        sorted_keys, return_inverse=True, return_counts=True
    )
    # Total number of unique voxels across the batch (before applying max_voxels limit)
    unique_total_voxels = counts.shape[0]

    # First sorted-point index of each unique voxel (XYZ) - for example, [0, 10, 30, ...], where 10
    # is the index of the first point in the second unique voxel, and 30 is the index
    # of the first point in the third unique voxel
    voxel_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]
    )

    # Compute max_voxels for each sample across voxels by ranking each voxel within its batch group.
    # For each sample within a batch, voxels are continuous since batch varies slowest in the key.
    voxel_batch_indices = sorted_batch_indices[voxel_starts]  # (unique_total_voxels,)

    # Compute unique batch indices and their counts. For example, if a batch has 3 samples, and the
    # number of voxels in each sample: [3, 2, 1], then voxel_batch_indices: [0, 0, 0, 1, 1, 2],
    # batch_local_id = [0, 0, 0, 1, 1, 2], batch_voxel_counts = [3, 2, 1].
    _, batch_local_id, batch_voxel_counts = torch.unique_consecutive(
        voxel_batch_indices, return_inverse=True, return_counts=True
    )
    # Compute the starting index of each batch's voxels. For example, if
    # batch_voxel_counts = [3, 2, 1], then batch_voxel_starts = [0, 3, 5],
    # which means the first batch's voxels start at index 0,
    # the second batch's voxels start at index 3, and the third batch's voxels start at index 5.
    batch_voxel_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), batch_voxel_counts.cumsum(0)[:-1]]
    )

    # Compute the global voxel ID for each voxel across a batch.
    voxel_global_id = torch.arange(unique_total_voxels, device=device)

    # Compute the local rank within each batch's voxels. For example, if
    # voxel_global_id = [0, 1, 2, 3, 4, 5] and batch_voxel_starts = [0, 3, 5], then
    # rank_in_batch = [0, 1, 2, 0, 1, 0]
    rank_in_batch = voxel_global_id - batch_voxel_starts[batch_local_id]

    # Keep only the first max_voxels voxels for each batch. For example, if max_voxels = 2, then
    # keep = [True, True, False, True, True, True], which means we keep the first two voxels of each
    # batch and discard the rest.
    # TODO(KokSeang): Consider to have a better way to handle the case when a batch has more than
    # max_voxels voxels, e.g., by randomly sampling or selecting the most populated voxels.
    keep_voxel: Bool[torch.Tensor, " unique_total_voxels"] = rank_in_batch < max_voxels

    # Compact the kept voxels, and build and old -> new voxel ID mapping. For example,
    # if keep_voxel = [True, True, False, True, True, True], then
    # voxel_id_mapping = [0, 1, -1, 2, 3, 4], which means the first voxel is kept and mapped to
    # new ID 0, the second voxel is kept and mapped to new ID 1, the third voxel is discarded and
    # mapped to -1, the fourth voxel is kept and mapped to new ID 2...
    kept_voxel_indices = torch.nonzero(keep_voxel, as_tuple=False).squeeze(1)
    num_voxels = kept_voxel_indices.shape[0]
    voxel_id_mapping = torch.full((unique_total_voxels,), -1, dtype=torch.long, device=device)
    voxel_id_mapping[kept_voxel_indices] = torch.arange(num_voxels, device=device)

    # Keep only the grid_coords and batch indices that belong to the kept voxels.
    kept_starts = voxel_starts[kept_voxel_indices]
    unique_coords = sorted_grid_coords[kept_starts]
    voxel_batch_indices = voxel_batch_indices[kept_voxel_indices]

    # Assign within-voxel slot to every point
    point_idx = torch.arange(sorted_keys.shape[0], device=device)
    # slot[i] = position of sorted point i within its voxel group
    slot = point_idx - voxel_starts[voxel_id]

    # Keep only the points that belong to the kept voxels and have slot < max_num_points.
    keep = keep_voxel[voxel_id] & (slot < max_num_points)
    # Compacted to [0, num_voxels)), it needs remapping because the valid voxel ids are not
    # continuous after filtering by per-sample max_voxels. For example,
    # if voxel_id = [0, 1, 2, 3, 4, 5] and keep_voxel = [True, True, False, True, True, True], then
    # voxel_id_mapping = [0, 1, -1, 2, 3, 4], voxel_id_mapping[voxel_id[keep]] = [0, 1, 2, 3, 4]
    valid_voxel_ids: Int64[torch.Tensor, " num_valid_voxels"] = voxel_id_mapping[voxel_id[keep]]
    valid_slots: Int64[torch.Tensor, " num_valid_voxels"] = slot[keep]
    valid_points = sorted_points[keep]

    # Fill voxels tensor via scatter
    voxels = points.new_zeros((num_voxels, max_num_points, channels))
    flat_idx = valid_voxel_ids * max_num_points + valid_slots
    voxels.view(-1, channels).scatter_(
        0,
        flat_idx.unsqueeze(1).expand(-1, channels),
        valid_points,
    )

    # Count valid points per voxel (capped at max_num_points)
    num_points = torch.zeros(num_voxels, dtype=torch.int32, device=device)
    num_points.scatter_add_(
        0,
        valid_voxel_ids,
        torch.ones(valid_voxel_ids.shape[0], dtype=torch.int32, device=device),
    )
    return VoxelsData(
        voxels=voxels,
        coords=unique_coords,
        num_points=num_points,
        batch_indices=voxel_batch_indices,
    )
