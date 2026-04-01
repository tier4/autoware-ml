"""Batching helpers for point-cloud tensors."""

from __future__ import annotations

import torch


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
