"""Point-cloud container structures."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.ops.indexing.operators import argsort
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.point_cloud.batching import offset_to_batch
from autoware_ml.utils.point_cloud.serialization.default import encode


def bit_length_tensor(x: torch.Tensor | int | float) -> torch.Tensor:
    """Return the integer bit length of a positive scalar tensor."""
    value = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    value = torch.clamp(value, min=1)
    return torch.floor(torch.log2(value)).to(torch.long) + 1


class Point(dict[str, torch.Tensor]):
    """Store point-cloud attributes in a dictionary-like container.

    The container exposes dictionary entries as attributes so point-cloud
    backbones can work with point features, coordinates, serialization
    metadata, and sparse-convolution views through one compact interface.
    """

    def __getattr__(self, key: str) -> torch.Tensor:
        """Expose dictionary entries as attributes."""
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: torch.Tensor) -> None:
        """Store attributes inside the underlying dictionary."""
        self[key] = value

    def serialization(
        self,
        order: Sequence[str],
        shuffle_orders: bool,
        depth: torch.Tensor | int | None = None,
    ) -> None:
        """Populate serialization fields required by point-cloud backbones.

        Args:
            order: Serialization order names applied to the point cloud.
            shuffle_orders: Whether to randomly permute serialization orders.
            depth: Optional fixed serialization depth. If omitted, derive it
                adaptively from the sample grid coordinates.
        """
        if "grid_coord" not in self:
            raise ValueError("grid_coord is required before point-cloud serialization.")
        if "offset" not in self:
            self["offset"] = torch.tensor(
                [self["coord"].shape[0]], device=self["coord"].device, dtype=torch.long
            )
        if "batch" not in self:
            self["batch"] = offset_to_batch(self["offset"], self["coord"])

        if depth is None:
            depth = bit_length_tensor(self["grid_coord"].max())
        elif isinstance(depth, torch.Tensor):
            depth = depth.to(device=self["coord"].device, dtype=torch.long)
        else:
            depth = torch.as_tensor(depth, device=self["coord"].device, dtype=torch.long)

        torch._assert(
            torch.all(depth * 3 + bit_length_tensor(self["offset"]) <= 63),
            "Point-cloud serialization exceeds supported int64 code range.",
        )
        torch._assert(
            torch.all(depth <= 16), "Point-cloud serialization depth exceeds supported range."
        )

        codes = [
            encode(self["grid_coord"], self["batch"], depth=depth, order=order_name)
            for order_name in order
        ]
        code = torch.stack(codes)
        serialized_order = argsort(code)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            1,
            serialized_order,
            torch.arange(code.shape[1], device=code.device).repeat(code.shape[0], 1),
        )
        if shuffle_orders:
            permutation = torch.randperm(code.shape[0], device=code.device)
            code = code[permutation]
            serialized_order = serialized_order[permutation]
            serialized_inverse = serialized_inverse[permutation]

        self["serialized_depth"] = depth
        self["serialized_code"] = code
        self["serialized_order"] = serialized_order
        self["serialized_inverse"] = serialized_inverse

    def sparsify(self, pad: int = 96) -> None:
        """Populate the sparse-convolution view of the point container.

        Args:
            pad: Spatial padding added to the sparse tensor shape.
        """
        if not IS_SPCONV_AVAILABLE:
            raise ModuleNotFoundError(
                "spconv is required for Point.sparsify() but is not installed."
            )

        import spconv.pytorch as spconv

        if "batch" not in self:
            self["batch"] = offset_to_batch(self["offset"], self["coord"])
        sparse_shape = torch.max(self["grid_coord"], dim=0).values + pad
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = spconv.SparseConvTensor(
            features=self["feat"],
            indices=torch.cat(
                [self["batch"].unsqueeze(-1).int(), self["grid_coord"].int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape.tolist(),
            batch_size=int(self["batch"][-1].item()) + 1 if self["batch"].numel() > 0 else 1,
        )
