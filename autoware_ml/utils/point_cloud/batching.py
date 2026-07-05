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

"""Batching helpers for point-cloud tensors."""

from __future__ import annotations

import torch


def infer_batch_size_from_voxel_coords(voxel_coords: torch.Tensor) -> int:
    """Infer batch size from batched voxel coordinates.

    Args:
        voxel_coords: Voxel coordinates with the batch index in column 0.

    Returns:
        Number of samples represented by the voxel coordinate tensor.
    """
    return int(voxel_coords[:, 0].max().item()) + 1 if voxel_coords.numel() > 0 else 1


@torch.inference_mode()
def offset_to_bincount(offset: torch.Tensor) -> torch.Tensor:
    """Convert cumulative offsets into per-batch counts.

    Args:
        offset: Cumulative batch offsets.

    Returns:
        Per-batch point counts.
    """
    if offset.numel() == 1:
        return offset
    return torch.cat([offset[:1], offset[1:] - offset[:-1]], dim=0)


@torch.inference_mode()
def offset_to_batch(offset: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
    """Expand cumulative offsets into per-point batch indices.

    Args:
        offset: Cumulative batch offsets.
        coords: Optional coordinate tensor used for singleton batches.

    Returns:
        Per-point batch indices.
    """
    if offset.numel() == 1 and coords is not None:
        return torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
    bincount = offset_to_bincount(offset)
    return torch.arange(bincount.numel(), device=offset.device, dtype=torch.long).repeat_interleave(
        bincount
    )


@torch.inference_mode()
def batch_to_offset(batch: torch.Tensor) -> torch.Tensor:
    """Convert per-point batch indices into cumulative offsets.

    Args:
        batch: Per-point batch indices.

    Returns:
        Cumulative batch offsets.
    """
    if batch.numel() == 0:
        return batch.new_zeros(0)
    return torch.cumsum(batch.bincount(), dim=0).long()
