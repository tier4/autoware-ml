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

"""Autograd bridges and ONNX symbolics for deployment-aware sparse ops."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from cumm import tensorview as tv
from spconv import constants
from spconv.algo import CONV_CPP
from spconv.constants import SPCONV_DO_SORT, SPCONV_USE_DIRECT_TABLE, AllocKeys
from spconv.core import ConvAlgo
from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.core_cc.csrc.sparse.convops.spops import ConvGemmOps
from spconv.pytorch import ops
from spconv.pytorch.core import ThrustSortAllocator
from spconv.pytorch.cppcore import (
    _TORCH_DTYPE_TO_TV,
    TorchAllocator,
    get_arch,
    get_current_stream,
    torch_tensor_to_tv,
)
from spconv.tools import CUDAKernelTimer
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes


def _kernel_volume(kernel_size: Sequence[int]) -> int:
    """Return the flattened kernel volume."""
    return int(np.prod(kernel_size))


def _set_symbolic_output_shape(value: torch.Tensor, sizes: Sequence[int | None]) -> None:
    """Attach an inferred tensor shape to an ONNX symbolic output when possible."""
    if hasattr(value.type(), "with_sizes"):
        value.setType(value.type().with_sizes(list(sizes)))


def _set_sparse_index_metadata(
    outputs: Sequence[torch.Tensor],
    indices: torch.Tensor,
    kernel_size: Sequence[int],
) -> None:
    """Attach inferred sparse-index output shapes to ONNX symbolic results."""
    indices_shape = _get_tensor_sizes(indices)
    if indices_shape is None:
        return

    kernel_volume = _kernel_volume(kernel_size)
    _set_symbolic_output_shape(outputs[0], [None, indices_shape[1]])
    _set_symbolic_output_shape(outputs[1], [2, kernel_volume, None])
    _set_symbolic_output_shape(outputs[2], [kernel_volume])
    _set_symbolic_output_shape(outputs[3], [])


def _set_sparse_feature_metadata(
    output: torch.Tensor,
    features: torch.Tensor,
    filters: torch.Tensor,
) -> None:
    """Attach inferred sparse-feature output shapes to an ONNX symbolic result."""
    feature_shape = _get_tensor_sizes(features)
    filter_shape = _get_tensor_sizes(filters)
    if feature_shape is None or filter_shape is None:
        return
    _set_symbolic_output_shape(output, [feature_shape[0], filter_shape[0]])


def _resolve_timer_handle(timer: CUDAKernelTimer) -> tv.CUDAKernelTimer:
    """Return the underlying CUDA timer used by low-level spconv kernels."""
    return timer._timer if timer._timer is not None else tv.CUDAKernelTimer(False)


def _current_stream_for(tensor: torch.Tensor) -> int:
    """Return the active CUDA stream for a tensor, or ``0`` on CPU."""
    return get_current_stream() if tensor.is_cuda else 0


def _as_int32_scalar_tensor(value: int, device: torch.device) -> torch.Tensor:
    """Wrap a scalar integer as a device-local ``int32`` tensor."""
    return torch.tensor([value], dtype=torch.int32, device=device)


def _validate_implicit_gemm_export_arguments(
    *,
    fp32_accum: bool | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor | None,
    output_add: torch.Tensor | None,
    output_dtype: torch.dtype | None,
) -> torch.dtype:
    """Validate the supported implicit-GEMM export argument subset."""
    if fp32_accum is not None:
        raise ValueError("Implicit-GEMM export does not support explicit fp32_accum overrides.")
    if bias is not None:
        raise ValueError("Implicit-GEMM export does not support fused bias.")
    if scale is not None:
        raise ValueError("Implicit-GEMM export does not support per-channel scaling.")
    if output_add is not None:
        raise ValueError("Implicit-GEMM export does not support fused residual addition.")
    if output_dtype is not None and output_dtype is not torch.float32:
        raise ValueError("Implicit-GEMM export supports only float32 outputs.")
    return torch.float32 if output_dtype is None else output_dtype


def _no_gradients(count: int) -> tuple[None, ...]:
    """Return a ``backward`` result that disables gradients for all inputs."""
    return (None,) * count


class GetIndicePairs(Function):
    """Bridge sparse indice-pair generation into autograd and ONNX export.

    The eager path delegates to spconv kernels, while the symbolic path emits
    the custom ONNX operator consumed by deployment tooling.
    """

    @staticmethod
    def symbolic(
        g,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: list[int],
        algo: ConvAlgo,
        ksize: list[int],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        out_padding: list[int],
        subm: bool,
        transpose: bool,
    ):
        """Register the ONNX symbolic for indice pair generation.

        Returns:
            ONNX outputs representing sparse output indices and pairing metadata.
        """
        outputs = g.op(
            "autoware::GetIndicePairs",
            indices,
            batch_size_i=batch_size,
            spatial_shape_i=spatial_shape,
            algo_i=algo.value,
            ksize_i=ksize,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            out_padding_i=out_padding,
            subm_i=subm,
            transpose_i=transpose,
            outputs=4,
        )
        _set_sparse_index_metadata(outputs, indices, ksize)
        return outputs

    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: list[int],
        algo: ConvAlgo,
        ksize: list[int],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        out_padding: list[int],
        subm: bool,
        transpose: bool,
    ) -> torch.Tensor:
        """Generate indice pairs during eager execution.

        Args:
            ctx: Autograd context.
            indices: Sparse indices tensor.
            batch_size: Batch size.
            spatial_shape: Sparse tensor spatial shape.
            algo: Sparse convolution algorithm.
            ksize: Kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            dilation: Convolution dilation.
            out_padding: Transposed-convolution output padding.
            subm: Whether the convolution is submanifold.
            transpose: Whether the convolution is transposed.

        Returns:
            Tuple-like tensor outputs consumed by the sparse-conv wrapper.
        """
        del ctx

        alloc = TorchAllocator(indices.device)
        stream = _current_stream_for(indices)

        num_act_out = SpconvOps.get_indice_pairs(
            alloc,
            torch_tensor_to_tv(indices),
            batch_size,
            spatial_shape,
            algo.value,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
            stream,
        )
        if subm:
            out_inds = indices
        else:
            out_inds = alloc.allocated[AllocKeys.OutIndices]

        pair = alloc.allocated[AllocKeys.PairFwd]
        indice_num_per_loc = alloc.allocated[AllocKeys.IndiceNumPerLoc]

        num_act_out = _as_int32_scalar_tensor(num_act_out, out_inds.device)

        return out_inds[:num_act_out], pair, indice_num_per_loc, num_act_out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return no gradients for indice pair generation.

        Args:
            ctx: Autograd context.
            grad_output: Upstream gradient tensor.

        Returns:
            Tuple of ``None`` gradients for all inputs.
        """
        del ctx, grad_output
        return _no_gradients(11)


class IndiceConvFunction(Function):
    """Bridge sparse indice convolution into autograd and ONNX export.

    The eager path uses spconv execution, while the symbolic path emits the
    custom convolution node expected by the deployment stack.
    """

    @staticmethod
    def symbolic(
        g,
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        algo,
        is_train,
        is_subm,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        bias: torch.Tensor | None = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
    ):
        """Register the ONNX symbolic for sparse indice convolution.

        Returns:
            ONNX value representing the sparse convolution output features.
        """

        output = g.op(
            "autoware::IndiceConv",
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            is_subm_i=is_subm,
            outputs=1,
        )

        _set_sparse_feature_metadata(output, features, filters)

        return output

    @staticmethod
    def forward(
        ctx,
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        algo,
        is_train: False,
        is_subm: True,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        bias: torch.Tensor | None = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
    ):
        """Execute sparse indice convolution during eager execution.

        Args:
            ctx: Autograd context.
            features: Sparse input features.
            filters: Sparse convolution filters.
            indice_pairs: Forward indice pairs.
            indice_pair_num: Number of pairs per kernel location.
            num_activate_out: Number of active output features.
            algo: Sparse convolution algorithm.
            is_train: Whether execution happens in training mode.
            is_subm: Whether the convolution is submanifold.
            timer: CUDA kernel timer.
            bias: Optional bias tensor.
            act_alpha: Activation alpha parameter.
            act_beta: Activation beta parameter.
            act_type: Activation type.

        Returns:
            Output sparse feature tensor.
        """
        del ctx

        if bias is not None:
            raise ValueError("Sparse indice convolution export does not support fused bias.")
        if act_alpha != 0.0 or act_beta != 0.0 or act_type != tv.gemm.Activation.None_:
            raise ValueError("Sparse indice convolution export does not support fused activations.")
        try:
            return ops.indice_conv(
                features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                is_train,
                is_subm,
                algo=algo,
                timer=timer,
                bias=bias,
                act_alpha=act_alpha,
                act_beta=act_beta,
                act_type=act_type,
            )
        except Exception as exc:
            context = (
                "[Exception|indice_conv|{}] feat={},w={},pair={},pairnum={},act={},algo={}".format(
                    "subm" if is_subm else "not_subm",
                    tuple(features.shape),
                    tuple(filters.shape),
                    tuple(indice_pairs.shape),
                    indice_pair_num,
                    num_activate_out,
                    algo,
                )
            )
            raise RuntimeError(context) from exc

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return no gradients for the deployment-only convolution path.

        Args:
            ctx: Autograd context.
            grad_output: Upstream gradient tensor.

        Returns:
            Tuple of ``None`` gradients for all inputs.
        """
        del ctx, grad_output
        return _no_gradients(13)


class GetIndicePairsImplicitGemm(Function):
    """Bridge implicit-GEMM indice generation into autograd and ONNX export.

    This variant mirrors spconv's implicit-GEMM path and exposes the matching
    metadata required by exportable sparse operators.
    """

    @staticmethod
    def symbolic(
        g,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: list[int],
        algo: ConvAlgo,
        ksize: list[int],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        out_padding: list[int],
        subm: bool,
        transpose: bool,
        is_train: bool,
        alloc: ThrustSortAllocator | None,
        timer: CUDAKernelTimer,
    ):
        """Register the ONNX symbolic for implicit-GEMM indice generation.

        Returns:
            ONNX outputs representing implicit-GEMM pairing metadata.
        """
        outputs = g.op(
            "autoware::GetIndicePairsImplicitGemm",
            indices,
            batch_size_i=batch_size,
            spatial_shape_i=spatial_shape,
            algo_i=algo.value,
            ksize_i=ksize,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            out_padding_i=out_padding,
            subm_i=subm,
            transpose_i=transpose,
            is_train_i=is_train,
            outputs=5,
        )
        indices_shape = _get_tensor_sizes(indices)
        if indices_shape is not None:
            kernel_volume = _kernel_volume(ksize)
            _set_symbolic_output_shape(outputs[0], [None, indices_shape[1]])
            _set_symbolic_output_shape(outputs[1], [kernel_volume, None])
            _set_symbolic_output_shape(outputs[2], [None, 1])
            _set_symbolic_output_shape(outputs[3], [None])
            _set_symbolic_output_shape(outputs[4], [])
        return outputs

    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: list[int],
        algo: ConvAlgo,
        ksize: list[int],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        out_padding: list[int],
        subm: bool,
        transpose: bool,
        is_train: bool,
        alloc: ThrustSortAllocator | None,
        timer: CUDAKernelTimer,
    ) -> torch.Tensor:
        """Generate implicit-GEMM indice pairs during eager execution.

        Args:
            ctx: Autograd context.
            indices: Sparse indices tensor.
            batch_size: Batch size.
            spatial_shape: Sparse tensor spatial shape.
            algo: Sparse convolution algorithm.
            ksize: Kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            dilation: Convolution dilation.
            out_padding: Transposed-convolution output padding.
            subm: Whether the convolution is submanifold.
            transpose: Whether the convolution is transposed.
            is_train: Whether execution happens in training mode.
            alloc: Optional thrust sort allocator.
            timer: CUDA kernel timer.

        Returns:
            Tuple of exported implicit-GEMM indice tensors.
        """
        del ctx

        num_out_act_bound: int = -1
        direct_table: bool = SPCONV_USE_DIRECT_TABLE
        do_sort = SPCONV_DO_SORT

        stream = _current_stream_for(indices)

        thalloc = TorchAllocator(indices.device)
        timer_cpp = _resolve_timer_handle(timer)

        mask_tensor, num_act_out = SpconvOps.get_indice_pairs_implicit_gemm(
            thalloc,
            torch_tensor_to_tv(indices),
            batch_size,
            spatial_shape,
            algo.value,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
            is_train,
            stream,
            num_out_act_bound,
            timer=timer_cpp,
            direct_table=direct_table,
            do_sort=do_sort,
        )

        num_act_out = _as_int32_scalar_tensor(num_act_out, indices.device)

        mask_split_count = mask_tensor.dim(0)
        if mask_split_count != 1:
            raise ValueError(
                f"Implicit-GEMM export supports exactly one mask split, got {mask_split_count}."
            )
        if subm:
            out_inds = indices
        else:
            out_inds = thalloc.allocated[AllocKeys.OutIndices]

        if subm:
            pair = thalloc.allocated[AllocKeys.PairFwd]
            pair_mask = thalloc.allocated[AllocKeys.PairMask]
            mask_argsort = thalloc.allocated[AllocKeys.MaskArgSort]
            return (out_inds, pair[0], pair_mask[0], mask_argsort[0], num_act_out)

        pair_fwd = thalloc.allocated[AllocKeys.PairFwd]
        pair_mask_fwd = thalloc.allocated[AllocKeys.PairMask]
        mask_argsort_fwd = thalloc.allocated[AllocKeys.MaskArgSort]
        return (out_inds, pair_fwd, pair_mask_fwd[0], mask_argsort_fwd[0], num_act_out)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return no gradients for indice generation.

        Args:
            ctx: Autograd context.
            grad_output: Upstream gradient tensor.

        Returns:
            Tuple of ``None`` gradients for all inputs.
        """
        del ctx, grad_output
        return _no_gradients(14)


class ImplicitGemm(Function):
    """Bridge implicit-GEMM sparse convolution into autograd and ONNX export.

    The function mirrors spconv's implicit-GEMM execution path and exposes a
    deployment-friendly symbolic node for exported graphs.
    """

    @staticmethod
    def symbolic(
        g,
        features: torch.Tensor,
        filters: torch.Tensor,
        pair_fwd: torch.Tensor,
        pair_mask_fwd_splits: torch.Tensor,
        mask_argsort_fwd_splits: torch.Tensor,
        num_activate_out: int,
        masks: list[npt.NDArray],
        is_train: bool,
        is_subm: bool,
        timer: CUDAKernelTimer,
        fp32_accum: bool | None,
        bias: torch.Tensor | None,
        act_alpha: float,
        act_beta: float,
        act_type: tv.gemm.Activation,
        output_scale: float,
        scale: torch.Tensor | None,
        output_add: torch.Tensor | None,
        output_add_scale: float,
        output_dtype: torch.dtype | None,
    ):
        """Register the ONNX symbolic for implicit-GEMM convolution.

        Returns:
            ONNX value representing the sparse convolution output features.
        """

        output = g.op(
            "autoware::ImplicitGemm",
            features,
            filters,
            pair_fwd,
            pair_mask_fwd_splits,
            mask_argsort_fwd_splits,
            is_train_i=is_train,
            is_subm_i=is_subm,
            fp32_accum_i=fp32_accum,
            act_alpha_f=act_alpha,
            act_beta_f=act_beta,
            output_scale_f=output_scale,
            output_add_scale_f=output_add_scale,
            outputs=1,
        )
        _set_sparse_feature_metadata(output, features, filters)

        return output

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        filters: torch.Tensor,
        pair_fwd: torch.Tensor,
        pair_mask_fwd_splits: torch.Tensor,
        mask_argsort_fwd_splits: torch.Tensor,
        num_activate_out: int,
        masks: list[npt.NDArray],
        is_train: bool,
        is_subm: bool,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        fp32_accum: bool | None = None,
        bias: torch.Tensor | None = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
        output_scale: float = 1.0,
        scale: torch.Tensor | None = None,
        output_add: torch.Tensor | None = None,
        output_add_scale: float = 0.0,
        output_dtype: torch.dtype | None = None,
    ):
        """Execute implicit-GEMM sparse convolution during eager execution.

        Args:
            ctx: Autograd context.
            features: Sparse input features.
            filters: Sparse convolution filters.
            pair_fwd: Forward indice pairs.
            pair_mask_fwd_splits: Forward pair mask splits.
            mask_argsort_fwd_splits: Forward sort indices for mask splits.
            num_activate_out: Number of active output features.
            masks: Kernel masks.
            is_train: Whether execution happens in training mode.
            is_subm: Whether the convolution is submanifold.
            timer: CUDA kernel timer.
            fp32_accum: Whether to accumulate in FP32.
            bias: Optional bias tensor.
            act_alpha: Activation alpha parameter.
            act_beta: Activation beta parameter.
            act_type: Activation type.
            output_scale: Output scaling factor.
            scale: Optional scale tensor.
            output_add: Optional tensor added to the output.
            output_add_scale: Scaling factor for ``output_add``.
            output_dtype: Requested output dtype.

        Returns:
            Output sparse feature tensor.
        """
        del ctx

        pair_mask_fwd_splits = [pair_mask_fwd_splits]
        mask_argsort_fwd_splits = [mask_argsort_fwd_splits]
        output_dtype = _validate_implicit_gemm_export_arguments(
            fp32_accum=fp32_accum,
            bias=bias,
            scale=scale,
            output_add=output_add,
            output_dtype=output_dtype,
        )

        stream = _current_stream_for(features)
        bias_tv = tv.Tensor()
        scale_tv = tv.Tensor()
        output_add_tv = tv.Tensor()

        if not features.is_contiguous():
            features = features.contiguous()
        if not filters.is_contiguous():
            filters = filters.contiguous()
        alloc = TorchAllocator(features.device, features.dtype == torch.qint8)
        features_tv = torch_tensor_to_tv(features)
        pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
        pair_mask_fwd_splits_tv = [torch_tensor_to_tv(t, tv.uint32) for t in pair_mask_fwd_splits]
        mask_argsort_fwd_splits_tv = [torch_tensor_to_tv(t) for t in mask_argsort_fwd_splits]

        filters_tv = torch_tensor_to_tv(filters)
        mask = np.array([np.iinfo(np.uint32).max], dtype=np.uint32)
        mask_tv = tv.from_numpy(mask).clone()
        timer_cpp = _resolve_timer_handle(timer)
        auto_fp32_accum = False
        fp32_accum = False
        arch = get_arch()
        output_dtype_tv = _TORCH_DTYPE_TO_TV[output_dtype]

        _, _ = ConvGemmOps.implicit_gemm(
            alloc,
            CONV_CPP,
            features_tv,
            filters_tv,
            pair_fwd_tv,
            pair_mask_fwd_splits_tv,
            mask_argsort_fwd_splits_tv,
            num_activate_out,
            mask_tv,
            arch,
            is_train,
            is_subm,
            stream,
            timer_cpp,
            auto_fp32_accum,
            fp32_accum,
            bias_tv,
            act_alpha,
            act_beta,
            act_type,
            use_tf32=constants.SPCONV_ALLOW_TF32,
            output_scale=output_scale,
            scale=scale_tv,
            output_add=output_add_tv,
            output_add_scale=output_add_scale,
            output_dtype=output_dtype_tv,
        )
        out_features = alloc.allocated[AllocKeys.OutFeatures]
        mask_output_fwd = alloc.allocated.get(AllocKeys.MaskOutputFwd, None)
        if is_train and mask_output_fwd is None:
            raise RuntimeError("Implicit-GEMM training path did not return mask_output_fwd.")

        return out_features

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return no gradients for the deployment-only implicit-GEMM path."""
        del ctx, grad_output
        return _no_gradients(20)


get_indice_pairs = GetIndicePairs.apply
indice_conv = IndiceConvFunction.apply


get_indice_pairs_implicit_gemm = GetIndicePairsImplicitGemm.apply
implicit_gemm = ImplicitGemm.apply
