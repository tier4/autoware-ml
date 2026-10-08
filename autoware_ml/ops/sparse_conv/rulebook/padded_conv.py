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

"""rulebook submanifold convolution that runs every channel width on compiled kernels.

rulebook_torch only has kernels for fp16 and bf16 operands and for a fixed set of
channel widths (multiples of 32). A layer with another width would silently take
the reference path, which keeps the gathered 27-neighbour tensor of every layer
alive for the backward pass (tens of GB for a PT-v3m3 frame).
:class:`PaddedSubMConv3d` zero-pads such layers to the next compiled width and
slices the result, so every width trains on real kernels.

This module imports ``rulebook_torch`` and is only imported once the rulebook
backend is selected, see :meth:`SparseConvBackend.submanifold_conv3d
<autoware_ml.ops.sparse_conv.backend.SparseConvBackend.submanifold_conv3d>`.
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import ClassVar, NamedTuple

import rulebook_torch
import torch
from rulebook_torch._ops import has_kernel, has_weight_grad_kernel, rulebook_for, subm_conv
from torch.nn import functional as F

from autoware_ml.ops.sparse_conv.tensor import SparseTensorLike

logger = logging.getLogger(__name__)


class PaddedWidths(NamedTuple):
    """Compiled channel widths a rulebook layer is zero-padded to.

    Attributes:
      in_channels: Padded input width.
      out_channels: Padded output width.
    """

    in_channels: int
    out_channels: int


class PaddedSubMConv3d(rulebook_torch.SubMConv3d):
    """rulebook submanifold convolution that pads channels to a compiled width.

    ``weight`` and ``bias`` keep the true ``[out, k, k, k, in]`` shape. When the
    true widths have kernels the layer is the plain rulebook layer. Otherwise the
    features and the weight are zero-padded to the nearest compiled widths, the
    real kernels run, and the surplus output channels are dropped; the padding
    and the slice are ordinary autograd ops, so the gradients of the true
    parameters are exact and the backward keeps only ``[N, padded_in]`` features
    per layer instead of the reference path's ``[N, 27, in]`` gather.

    Attributes:
      PADDING_CANDIDATES: Channel widths tried, in order, when a layer's own width
        has no compiled kernel. rulebook ships tiles for multiples of 32; the kernel
        registry is queried, not assumed.
    """

    PADDING_CANDIDATES: ClassVar[tuple[int, ...]] = tuple(range(32, 1025, 32))
    _fp32_warned: ClassVar[bool] = False

    @staticmethod
    def has_kernels(
        in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], dtype: torch.dtype
    ) -> bool:
        """Return whether rulebook has a forward and a weight-gradient kernel for a layer.

        Args:
            in_channels: Input width of the layer.
            out_channels: Output width of the layer.
            kernel_size: Cubic kernel size as a triple.
            dtype: Compute dtype of the operands.

        Returns:
            ``True`` when both kernels are compiled.
        """
        return bool(
            has_kernel(in_channels, out_channels, kernel_size, dtype)
            and has_weight_grad_kernel(in_channels, out_channels, dtype)
        )

    @staticmethod
    @functools.cache
    def padded_widths(
        in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], dtype: torch.dtype
    ) -> PaddedWidths | None:
        """Return the compiled channel widths a layer is padded to, if needed.

        The padded input widths are searched in the outer loop and the padded output
        widths in the inner one, so the smallest padded input width wins first.

        Args:
            in_channels: True input width of the layer.
            out_channels: True output width of the layer.
            kernel_size: Cubic kernel size as a triple.
            dtype: Compute dtype of the operands (the autocast dtype under autocast).

        Returns:
            ``None`` when the layer's own widths have a forward and a weight-gradient
            kernel (or when the layer has no padded alternative either), otherwise the
            first padded widths that have both.
        """
        if PaddedSubMConv3d.has_kernels(in_channels, out_channels, kernel_size, dtype):
            return None
        candidates = PaddedSubMConv3d.PADDING_CANDIDATES
        widths = next(
            (
                PaddedWidths(padded_in, padded_out)
                for padded_in, padded_out in itertools.product(
                    [width for width in candidates if width >= in_channels],
                    [width for width in candidates if width >= out_channels],
                )
                if PaddedSubMConv3d.has_kernels(padded_in, padded_out, kernel_size, dtype)
            ),
            None,
        )
        PaddedSubMConv3d._log_padded_widths(in_channels, out_channels, kernel_size, dtype, widths)
        return widths

    @staticmethod
    def _log_padded_widths(
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        dtype: torch.dtype,
        widths: PaddedWidths | None,
    ) -> None:
        """Log how a layer without kernels for its own widths runs.

        Args:
            in_channels: True input width of the layer.
            out_channels: True output width of the layer.
            kernel_size: Cubic kernel size as a triple.
            dtype: Compute dtype of the operands.
            widths: Padded widths found, or ``None`` when no padded width fits.
        """
        dtype_name = str(dtype).replace("torch.", "")
        if widths is not None:
            logger.info(
                "rulebook has no %s kernel for a %d->%d submanifold convolution; the layer runs "
                "zero-padded as %d->%d.",
                dtype_name,
                in_channels,
                out_channels,
                *widths,
            )
            return
        logger.warning(
            "rulebook has no %s kernel for a %d->%d submanifold convolution with kernel %s "
            "and no padded width fits either; the layer runs on the slow reference path.",
            dtype_name,
            in_channels,
            out_channels,
            kernel_size,
        )

    @classmethod
    def _warn_if_fp32(cls, features: torch.Tensor) -> None:
        """Warn once when a layer is about to run without a compiled kernel.

        rulebook only ships fp16/bf16 kernels; fp32 operands outside autocast take a
        plain-PyTorch gather-and-matmul path that is correct but roughly an order of
        magnitude slower. A training run that lands there almost always forgot
        ``trainer.precision=16-mixed``.

        Args:
            features: Input features of the layer.
        """
        runs_fp32 = (
            features.is_cuda
            and features.dtype == torch.float32
            and not torch.is_autocast_enabled("cuda")
        )
        if cls._fp32_warned or not runs_fp32:
            return
        cls._fp32_warned = True
        logger.warning(
            "rulebook_torch.SubMConv3d received fp32 features outside autocast and falls back "
            "to its slow reference path. Train with trainer.precision=16-mixed, or set "
            "sparse_conv_backend=spconv for fp32 runs."
        )

    @staticmethod
    def _compute_dtype(features: torch.Tensor) -> torch.dtype:
        """Return the dtype the kernels see: the autocast dtype under autocast.

        Args:
            features: Input features of the layer.

        Returns:
            Compute dtype of the operands.
        """
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return features.dtype

    def _padded_weight(self, widths: PaddedWidths) -> torch.Tensor:
        """Return the ``[out, k, k, k, in]`` weight zero-padded to the given widths.

        Args:
            widths: Padded widths of the layer.

        Returns:
            The padded weight, or the weight itself when no padding is needed.
        """
        extra_in = widths.in_channels - self.in_channels
        extra_out = widths.out_channels - self.out_channels
        if extra_in <= 0 and extra_out <= 0:
            return self.weight
        # F.pad takes (before, after) pairs from the last dimension backwards:
        # in, the three kernel axes, out.
        no_padding = (0, 0)
        pad = (0, extra_in) + no_padding * 3 + (0, extra_out)
        return F.pad(self.weight, pad)

    def forward(self, x: SparseTensorLike) -> SparseTensorLike:
        """Convolve a sparse tensor, padding the channels when the widths need it.

        Args:
            x: Sparse tensor holding the input features.

        Returns:
            Sparse tensor holding the convolved features.
        """
        features = x.features
        self._warn_if_fp32(features)
        if features.shape[0] == 0 or not features.is_cuda:
            return super().forward(x)
        widths = self.padded_widths(
            self.in_channels, self.out_channels, self.kernel_size, self._compute_dtype(features)
        )
        if widths is None:
            return super().forward(x)
        rb = rulebook_for(x, self.indice_key, self.kernel_size)
        if widths.in_channels > self.in_channels:
            features = F.pad(features, (0, widths.in_channels - self.in_channels))
        out = subm_conv(features.contiguous(), self._padded_weight(widths), None, rb)
        if widths.out_channels > self.out_channels:
            out = out[:, : self.out_channels]
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return x.replace_feature(out)
