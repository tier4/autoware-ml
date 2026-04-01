# Copyright 2025 TIER IV, Inc.
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

"""BEV pooling operations for bird's-eye view feature extraction.

This module provides CUDA-accelerated operations for pooling image features
into a bird's-eye view (BEV) representation, commonly used in 3D perception
tasks for autonomous driving.
"""

from __future__ import annotations

import torch

from autoware_ml.ops.bev_pool import bev_pool_ext


def _compute_intervals(ranks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interval starts and lengths for sorted rank values.

    Args:
        ranks: Sorted rank tensor used by BEV pooling.

    Returns:
        Tuple of interval starts and interval lengths.
    """
    if ranks.numel() == 0:
        raise ValueError("BEV pooling requires at least one projected feature.")
    kept = torch.ones(ranks.shape[0], device=ranks.device, dtype=torch.bool)
    kept[1:] = ranks[1:] != ranks[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks.shape[0] - interval_starts[-1]
    return interval_starts, interval_lengths


class QuickCumsumTrainingCuda(torch.autograd.Function):
    """CUDA-accelerated cumulative sum for BEV pooling during training.

    This function computes interval-based pooling of features into a BEV grid,
    with full backward pass support for gradient computation.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        geom_feats: torch.Tensor,
        ranks: torch.Tensor,
        B: int,
        D: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute forward pass of BEV pooling.

        Args:
            ctx: Autograd context for saving tensors.
            x: Input features of shape (N, C) where N is number of points.
            geom_feats: Geometric features/coordinates of shape (N, 4).
            ranks: Rank indices for sorting/grouping points of shape (N,).
            B: Batch size.
            D: Depth dimension of output BEV grid.
            H: Height dimension of output BEV grid.
            W: Width dimension of output BEV grid.

        Returns:
            Pooled BEV features of shape (B, D, H, W, C).
        """
        interval_starts, interval_lengths = _compute_intervals(ranks)
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        out_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        """Compute backward pass for gradient propagation.

        Args:
            ctx: Autograd context with saved tensors.
            out_grad: Gradient of loss w.r.t. output, shape (B, D, H, W, C).

        Returns:
            Tuple of gradients for each input (only x_grad is non-None).
        """
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


class QuickCumsumCuda(torch.autograd.Function):
    """CUDA-accelerated cumulative sum for BEV pooling during inference.

    This function is optimized for inference with ONNX export support.
    It does not support backward pass computation.
    """

    @staticmethod
    def symbolic(
        g: torch.Graph,
        x: torch.Tensor,
        geom_feats: torch.Tensor,
        interval_lengths: torch.Tensor,
        interval_starts: torch.Tensor,
        B: int,
        D: int,
        H: int,
        W: int,
    ) -> torch.Value:
        """Define ONNX symbolic representation for export.

        Args:
            g: ONNX graph being constructed.
            x: Input features tensor.
            geom_feats: Geometric features/coordinates tensor.
            interval_lengths: Length of each pooling interval.
            interval_starts: Starting index of each pooling interval.
            B: Batch size.
            D: Depth dimension.
            H: Height dimension.
            W: Width dimension.

        Returns:
            ONNX graph node representing this operation.
        """
        return g.op(
            "autoware::QuickCumsumCuda",
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            batch_size_i=B,
            dimension_i=D,
            height_i=H,
            width_i=W,
            outputs=1,
        )

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        geom_feats: torch.Tensor,
        interval_lengths: torch.Tensor,
        interval_starts: torch.Tensor,
        B: int,
        D: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute forward pass of BEV pooling (inference mode).

        Args:
            ctx: Autograd context (unused in inference).
            x: Input features of shape (N, C).
            geom_feats: Geometric features/coordinates of shape (N, 4).
            interval_lengths: Length of each pooling interval.
            interval_starts: Starting index of each pooling interval.
            B: Batch size.
            D: Depth dimension of output BEV grid.
            H: Height dimension of output BEV grid.
            W: Width dimension of output BEV grid.

        Returns:
            Pooled BEV features of shape (B, D, H, W, C).
        """
        return bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        out_grad: torch.Tensor,
    ) -> tuple[None, ...]:
        """Backward pass is not supported for inference-only function.

        Raises:
            NotImplementedError: Always raised as this is inference-only.
        """
        raise NotImplementedError("QuickCumsumCuda does not support backward pass")


def bev_pool(
    feats: torch.Tensor,
    coords: torch.Tensor,
    ranks: torch.Tensor,
    B: int,
    D: int,
    H: int,
    W: int,
    is_training: bool,
) -> torch.Tensor:
    """Perform BEV (Bird's Eye View) pooling operation.

    This function pools image features into a bird's-eye view grid representation.
    It supports both training mode (with gradient computation) and inference mode
    (optimized for ONNX export).

    Args:
        feats: Input features of shape (N, C) where N is the number of points
            and C is the number of channels.
        coords: Coordinates mapping points to BEV grid positions, shape (N, 4).
            Format is (height_idx, width_idx, depth_idx, batch_idx).
        ranks: Rank indices used for sorting and grouping points, shape (N,).
        B: Batch size.
        D: Depth dimension of the output BEV grid.
        H: Height dimension of the output BEV grid.
        W: Width dimension of the output BEV grid.
        is_training: If True, uses training path with gradient support.
            If False, uses inference path optimized for ONNX export.

    Returns:
        Pooled BEV features of shape (B, C, D, H, W).

    Raises:
        AssertionError: If feats and coords have mismatched first dimensions.
    """
    if is_training:
        x = QuickCumsumTrainingCuda.apply(feats, coords, ranks, B, D, H, W)
    else:
        # Compute intervals outside the autograd function for ONNX tracing
        interval_starts, interval_lengths = _compute_intervals(ranks)

        if coords.dtype != torch.int32:
            coords = coords.int()

        x = QuickCumsumCuda.apply(
            feats,
            coords,
            interval_lengths,
            interval_starts,
            int(B),
            int(D),
            int(H),
            int(W),
        )

    # Permute from (B, D, H, W, C) to (B, C, D, H, W)
    return x.permute(0, 4, 1, 2, 3).contiguous()
