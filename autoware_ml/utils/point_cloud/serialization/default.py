"""Default serialization helpers for point-cloud models."""

from __future__ import annotations

import torch

from autoware_ml.utils.point_cloud.serialization.hilbert import decode as hilbert_decode_
from autoware_ml.utils.point_cloud.serialization.hilbert import encode as hilbert_encode_
from autoware_ml.utils.point_cloud.serialization.z_order import key2xyz as z_order_decode_
from autoware_ml.utils.point_cloud.serialization.z_order import xyz2key as z_order_encode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """Encode voxel coordinates into serialized integer codes."""
    if order not in {"z", "z-trans", "hilbert", "hilbert-trans"}:
        raise ValueError(f"Unsupported serialization order: {order}")
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    """Decode serialized integer codes into voxel coordinates and batch ids."""
    if order not in {"z", "hilbert"}:
        raise ValueError(f"Unsupported serialization order: {order}")
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """Encode coordinates with Morton ordering."""
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    return z_order_encode_(x, y, z, b=None, depth=depth)


def z_order_decode(code: torch.Tensor, depth):
    """Decode Morton-ordered coordinates."""
    x, y, z, _ = z_order_decode_(code, depth=depth)
    return torch.stack([x, y, z], dim=-1)


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    """Encode coordinates with Hilbert ordering."""
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    """Decode Hilbert-ordered coordinates."""
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
