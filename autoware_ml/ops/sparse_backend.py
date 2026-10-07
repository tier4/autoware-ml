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

"""Backend selection for the submanifold sparse convolutions of the PTv3 family.

Every sparse convolution of PTv3 and LitePT is a stride-1 submanifold convolution:
the conditional positional encoding at the front of each block and, optionally,
the embedding stem. Two libraries implement it behind the same ``SubMConv3d``
call, weight layout and ``indice_key`` caching:

* ``spconv`` - the historical dependency. Its kernels accumulate in fp16 and are
  not registered with autocast, so a ``16-mixed`` run either aborts in the kernel
  tuner or trains on fp16 partial sums. It stays required for the strided
  detection encoders and for ONNX export.
* ``rulebook_torch`` - the in-house replacement developed by Max Schmeller (repository
  ``spconv-replacement``); this module only adapts PTv3 to it.
  Every kernel accumulates in fp32 and the autograd function casts inside, so
  mixed precision reproduces fp32 training. It only has kernels for fp16 and
  bf16 operands; an fp32 forward runs through a slow reference path, so it is
  meant for mixed precision (``16-mixed`` or ``bf16-mixed``). Its kernels also exist only for a
  fixed set of channel widths (multiples of 32); a layer with another width would
  silently take the same reference path, which keeps the gathered 27-neighbour
  tensor of every layer alive for the backward pass (tens of GB for a PT-v3m3
  frame). :class:`PaddedSubMConv3d` zero-pads such layers to the next compiled
  width and slices the result, so every width trains on real kernels.

Models take a ``sparse_conv_backend`` argument (``"auto"``, ``"spconv"`` or
``"rulebook"``) and build their layers through :func:`submanifold_conv3d`, so the
choice is one config key and the rest of the code never imports either library at
module level. The configs default to ``"spconv"``, the historical behaviour; a mixed
precision recipe selects ``"rulebook"`` (``configs/defaults/amp_rulebook.yaml``), and
``"auto"`` takes rulebook whenever it is installed. rulebook is CUDA-only, so CPU paths
(CPU unit tests, CPU inference) keep ``"spconv"``.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

logger = logging.getLogger(__name__)

IS_RULEBOOK_AVAILABLE = find_spec("rulebook_torch") is not None

SPARSE_CONV_BACKENDS = ("auto", "spconv", "rulebook")


def resolve_sparse_conv_backend(backend: str = "auto") -> str:
    """Resolve a backend request against the installed libraries.

    Args:
        backend: ``"auto"``, ``"spconv"`` or ``"rulebook"``.

    Returns:
        ``"spconv"`` or ``"rulebook"``.

    Raises:
        ValueError: Raised for an unknown backend name.
        ModuleNotFoundError: Raised when the requested backend, or under
            ``"auto"`` either backend, is not installed.
    """
    if backend not in SPARSE_CONV_BACKENDS:
        raise ValueError(
            f"Unknown sparse convolution backend {backend!r}, expected one of "
            f"{SPARSE_CONV_BACKENDS}."
        )
    if backend == "auto":
        if IS_RULEBOOK_AVAILABLE:
            return "rulebook"
        if IS_SPCONV_AVAILABLE:
            return "spconv"
        raise ModuleNotFoundError(
            "Neither 'rulebook_torch' nor 'spconv' is installed; the PTv3 sparse "
            "convolutions need one of them."
        )
    if backend == "rulebook" and not IS_RULEBOOK_AVAILABLE:
        raise ModuleNotFoundError(
            "sparse_conv_backend='rulebook' requested but 'rulebook_torch' is not installed."
        )
    if backend == "spconv" and not IS_SPCONV_AVAILABLE:
        raise ModuleNotFoundError(
            "sparse_conv_backend='spconv' requested but 'spconv' is not installed."
        )
    return backend


# Channel widths tried, in order, when a layer's own width has no compiled kernel.
# rulebook ships tiles for multiples of 32; the registry is queried, not assumed.
_PADDING_CANDIDATES = tuple(range(32, 1025, 32))
_padded_widths_cache: dict[
    tuple[int, int, tuple[int, int, int], torch.dtype], tuple[int, int] | None
] = {}
_padded_layer_class: type[nn.Module] | None = None


def rulebook_padded_widths(
    in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], dtype: torch.dtype
) -> tuple[int, int] | None:
    """Return the compiled channel widths a rulebook layer is padded to, if needed.

    Args:
        in_channels: True input width of the layer.
        out_channels: True output width of the layer.
        kernel_size: Cubic kernel size as a triple.
        dtype: Compute dtype of the operands (the autocast dtype under autocast).

    Returns:
        ``None`` when the layer's own widths have a forward and a weight-gradient
        kernel (or when the stem-like layer has no padded alternative either),
        otherwise the smallest ``(padded_in, padded_out)`` pair that has both.
    """
    key = (in_channels, out_channels, kernel_size, dtype)
    if key in _padded_widths_cache:
        return _padded_widths_cache[key]
    from rulebook_torch._ops import has_kernel, has_weight_grad_kernel

    def supported(c_in: int, c_out: int) -> bool:
        return bool(
            has_kernel(c_in, c_out, kernel_size, dtype)
            and has_weight_grad_kernel(c_in, c_out, dtype)
        )

    result: tuple[int, int] | None = None
    if not supported(in_channels, out_channels):
        ins = [c for c in _PADDING_CANDIDATES if c >= in_channels]
        outs = [c for c in _PADDING_CANDIDATES if c >= out_channels]
        for padded_in in ins:
            for padded_out in outs:
                if supported(padded_in, padded_out):
                    result = (padded_in, padded_out)
                    break
            if result is not None:
                break
        if result is not None:
            logger.info(
                "rulebook has no %s kernel for a %d->%d submanifold convolution; the layer runs "
                "zero-padded as %d->%d.",
                str(dtype).replace("torch.", ""),
                in_channels,
                out_channels,
                *result,
            )
        else:
            logger.warning(
                "rulebook has no %s kernel for a %d->%d submanifold convolution with kernel %s "
                "and no padded width fits either; the layer runs on the slow reference path.",
                str(dtype).replace("torch.", ""),
                in_channels,
                out_channels,
                kernel_size,
            )
    _padded_widths_cache[key] = result
    return result


def padded_rulebook_layer_class() -> type[nn.Module]:
    """Return the rulebook ``SubMConv3d`` subclass that pads unsupported widths.

    The class is created on first use so that this module imports without
    rulebook installed. It is a subclass, so every ``isinstance`` check, the
    export conversion and the state dict layout of the plain layer apply.

    Returns:
        The layer class.
    """
    global _padded_layer_class
    if _padded_layer_class is not None:
        return _padded_layer_class
    import rulebook_torch
    from rulebook_torch._ops import rulebook_for, subm_conv

    class PaddedSubMConv3d(rulebook_torch.SubMConv3d):
        """rulebook submanifold convolution that pads channels to a compiled width.

        ``weight`` and ``bias`` keep the true ``[out, k, k, k, in]`` shape. When the
        true widths have kernels the layer is the plain rulebook layer. Otherwise the
        features and the weight are zero-padded to the nearest compiled widths, the
        real kernels run, and the surplus output channels are dropped; the padding
        and the slice are ordinary autograd ops, so the gradients of the true
        parameters are exact and the backward keeps only ``[N, padded_in]`` features
        per layer instead of the reference path's ``[N, 27, in]`` gather.
        """

        def forward(self, x: Any) -> Any:
            features = x.features
            if features.shape[0] == 0 or not features.is_cuda:
                return super().forward(x)
            dtype = features.dtype
            if torch.is_autocast_enabled("cuda"):
                dtype = torch.get_autocast_dtype("cuda")
            widths = rulebook_padded_widths(
                self.in_channels, self.out_channels, self.kernel_size, dtype
            )
            if widths is None:
                return super().forward(x)
            padded_in, padded_out = widths
            rb = rulebook_for(x, self.indice_key, self.kernel_size)
            if padded_in > self.in_channels:
                features = F.pad(features, (0, padded_in - self.in_channels))
            weight = self.weight
            if padded_in > self.in_channels or padded_out > self.out_channels:
                weight = F.pad(
                    weight,
                    (
                        0,
                        padded_in - self.in_channels,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        padded_out - self.out_channels,
                    ),
                )
            out = subm_conv(features.contiguous(), weight, None, rb)
            if padded_out > self.out_channels:
                out = out[:, : self.out_channels]
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return x.replace_feature(out)

    _padded_layer_class = PaddedSubMConv3d
    return PaddedSubMConv3d


def submanifold_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    bias: bool,
    indice_key: str,
    backend: str = "auto",
) -> nn.Module:
    """Build a stride-1 submanifold convolution on the requested backend.

    Both backends share the KRSC weight layout ``[out, k, k, k, in]`` and the
    parameter names ``weight``/``bias``, so checkpoints trained on one load on the
    other unchanged.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        kernel_size: Cubic kernel size.
        bias: Whether the layer carries a bias.
        indice_key: Key under which the layer caches its index pairs (spconv) or
            rulebook on the sparse tensor, shared by the layers of one stage.
        backend: Backend request, see :func:`resolve_sparse_conv_backend`.

    Returns:
        The convolution module.
    """
    resolved = resolve_sparse_conv_backend(backend)
    if resolved == "rulebook":
        return padded_rulebook_layer_class()(
            in_channels, out_channels, kernel_size=kernel_size, bias=bias, indice_key=indice_key
        )
    import spconv.pytorch as spconv

    return spconv.SubMConv3d(
        in_channels, out_channels, kernel_size=kernel_size, bias=bias, indice_key=indice_key
    )


def sparse_conv_backend_of(module: nn.Module) -> str | None:
    """Return the backend a sparse convolution module belongs to.

    Args:
        module: Module inspected.

    Returns:
        ``"spconv"``, ``"rulebook"``, or ``None`` when the module is not a sparse
        convolution of either backend.
    """
    if IS_RULEBOOK_AVAILABLE:
        import rulebook_torch

        if isinstance(module, rulebook_torch.SubMConv3d):
            return "rulebook"
    if IS_SPCONV_AVAILABLE:
        import spconv.pytorch as spconv

        if spconv.modules.is_spconv_module(module):
            return "spconv"
    return None


def submanifold_conv_backend_of(module: nn.Module) -> str | None:
    """Return the backend of a stride-1 submanifold convolution layer.

    Unlike :func:`sparse_conv_backend_of`, which recognises every spconv module
    (containers and strided layers included), this only matches the
    ``SubMConv3d`` classes the PTv3 family builds.

    Args:
        module: Module inspected.

    Returns:
        ``"spconv"``, ``"rulebook"``, or ``None``.
    """
    if IS_RULEBOOK_AVAILABLE:
        import rulebook_torch

        if isinstance(module, rulebook_torch.SubMConv3d):
            return "rulebook"
    if IS_SPCONV_AVAILABLE:
        import spconv.pytorch as spconv

        if isinstance(module, spconv.SubMConv3d):
            return "spconv"
    return None


def is_sparse_conv_module(module: nn.Module) -> bool:
    """Return whether a module consumes sparse convolution tensors.

    Args:
        module: Module inspected for sparse-convolution semantics.

    Returns:
        ``True`` when the module expects sparse convolution tensors.
    """
    return sparse_conv_backend_of(module) is not None


def is_sparse_conv_tensor(value: Any) -> bool:
    """Return whether a value is the sparse tensor container of either backend.

    Args:
        value: Value inspected.

    Returns:
        ``True`` for ``spconv.SparseConvTensor`` and ``rulebook_torch.SparseTensor``.
    """
    if IS_SPCONV_AVAILABLE:
        import spconv.pytorch as spconv

        if isinstance(value, spconv.SparseConvTensor):
            return True
    if IS_RULEBOOK_AVAILABLE:
        import rulebook_torch

        if isinstance(value, rulebook_torch.SparseTensor):
            return True
    return False


def sparse_conv_tensor(
    features: torch.Tensor,
    indices: torch.Tensor,
    spatial_shape: list[int],
    batch_size: int,
) -> Any:
    """Build the sparse tensor container the installed backends consume.

    spconv's ``SparseConvTensor`` is used whenever spconv is installed, because
    rulebook layers accept it as well and the export wrappers require it. Without
    spconv the rulebook container, which carries the same four fields plus the
    ``indice_dict`` cache, is used instead.

    Args:
        features: Per voxel features ``[N, C]``.
        indices: Integer ``[N, 4]`` tensor of batch index and grid coordinate.
        spatial_shape: Dense grid extent per axis.
        batch_size: Number of samples in the batch.

    Returns:
        The sparse tensor container.

    Raises:
        ModuleNotFoundError: Raised when neither backend is installed.
    """
    if IS_SPCONV_AVAILABLE:
        import spconv.pytorch as spconv

        return spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
    if IS_RULEBOOK_AVAILABLE:
        import rulebook_torch

        return rulebook_torch.SparseTensor(
            features, indices, spatial_shape=spatial_shape, batch_size=batch_size
        )
    raise ModuleNotFoundError(
        "Building a sparse tensor requires 'spconv' or 'rulebook_torch'; neither is installed."
    )


_RULEBOOK_FP32_WARNED = False


def warn_if_rulebook_runs_fp32(module: nn.Module, features: torch.Tensor) -> None:
    """Warn once when a rulebook layer is about to run without a compiled kernel.

    rulebook only ships fp16/bf16 kernels; fp32 operands outside autocast take a
    plain-PyTorch gather-and-matmul path that is correct but roughly an order of
    magnitude slower. A training run that lands there almost always forgot
    ``trainer.precision=16-mixed``.

    Args:
        module: The sparse convolution about to run.
        features: Its input features.
    """
    global _RULEBOOK_FP32_WARNED
    if _RULEBOOK_FP32_WARNED or not features.is_cuda or features.dtype != torch.float32:
        return
    if torch.is_autocast_enabled("cuda"):
        return
    if sparse_conv_backend_of(module) != "rulebook":
        return
    _RULEBOOK_FP32_WARNED = True
    logger.warning(
        "rulebook_torch.SubMConv3d received fp32 features outside autocast and falls back to "
        "its slow reference path. Train with trainer.precision=16-mixed, or set "
        "sparse_conv_backend=spconv for fp32 runs."
    )
