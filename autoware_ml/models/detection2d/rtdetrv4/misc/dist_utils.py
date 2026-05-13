"""Minimal distributed helpers required by the vendored RT-DETR criterion."""

from __future__ import annotations

import torch.distributed


def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def barrier() -> None:
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
