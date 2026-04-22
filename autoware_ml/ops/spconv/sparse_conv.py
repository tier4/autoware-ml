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

"""Sparse-convolution layer wrappers used by deployment-aware sparse models."""

from dataclasses import dataclass
from typing import Any

import torch
from cumm import tensorview as tv
from spconv.core import ConvAlgo
from spconv.debug_utils import spconv_save_debug_data
from spconv.pytorch import ops
from spconv.pytorch.conv import SparseConvolution as SparseConvolutionBase
from spconv.pytorch.core import ImplicitGemmIndiceData, IndiceData, SparseConvTensor
from spconv.utils import nullcontext
from torch.nn import functional as F

import autoware_ml.ops.spconv.sparse_functional as Fsp_custom

_MAX_NUM_VOXELS_DURING_TRAINING = "max_num_voxels_during_training"


def _apply_activation(
    x: torch.Tensor,
    act_type: tv.gemm.Activation,
    act_alpha: float,
    act_beta: float,
) -> torch.Tensor:
    """Apply the configured activation to a tensor.

    Args:
        x: Input tensor.
        act_type: Activation type.
        act_alpha: Activation alpha parameter.
        act_beta: Activation beta parameter.

    Returns:
        Activated tensor.
    """
    del act_beta
    if act_type == tv.gemm.Activation.None_:
        return x
    if act_type == tv.gemm.Activation.ReLU:
        return F.relu(x)
    if act_type == tv.gemm.Activation.Sigmoid:
        return torch.sigmoid(x)
    if act_type == tv.gemm.Activation.LeakyReLU:
        return F.leaky_relu(x, act_alpha)
    raise NotImplementedError(f"Unsupported sparse activation type: {act_type!r}")


def _resolve_output_spatial_shape(
    spatial_shape: list[int],
    *,
    subm: bool,
    transposed: bool,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    output_padding: list[int],
) -> list[int]:
    """Resolve the sparse spatial shape produced by a convolution layer."""
    if subm:
        return spatial_shape
    if transposed:
        return ops.get_deconv_output_size(
            spatial_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
        )
    return ops.get_conv_output_size(
        spatial_shape,
        kernel_size,
        stride,
        padding,
        dilation,
    )


def _raise_generation_error(exc: Exception, *, kind: str, **context: Any) -> None:
    """Raise a contextualized sparse-convolution export error."""
    message = f"[Exception|{kind}] " + ",".join(f"{key}={value}" for key, value in context.items())
    raise RuntimeError(message) from exc


@dataclass(frozen=True)
class _NativeExecutionPlan:
    """Resolved sparse metadata for the native spconv execution path."""

    outids: torch.Tensor
    indice_pairs: torch.Tensor
    indice_pair_num: torch.Tensor
    num_act_out: int | torch.Tensor
    out_spatial_shape: list[int]


@dataclass(frozen=True)
class _ImplicitGemmExecutionPlan:
    """Resolved sparse metadata for the implicit-GEMM execution path."""

    outids: torch.Tensor
    pair_fwd: torch.Tensor
    pair_mask_fwd_splits: torch.Tensor
    mask_argsort_fwd_splits: torch.Tensor
    masks: list[Any]
    num_act_out: int | torch.Tensor
    out_spatial_shape: list[int]


class SparseConvolution(SparseConvolutionBase):
    """Wrap spconv sparse convolutions with deployment-oriented execution paths.

    The wrapper preserves eager spconv behavior while routing export through
    the custom ONNX-friendly sparse functional bridges defined by autoware-ml.
    """

    def _validate_export_configuration(
        self,
        input_tensor: SparseConvTensor,
        weight: torch.Tensor,
        *,
        training: bool,
    ) -> None:
        """Validate the subset of sparse-convolution behavior supported here."""
        if input_tensor.is_quantized and weight.is_quantized:
            raise NotImplementedError("INT8 sparse-convolution export is not supported.")
        if training:
            raise NotImplementedError("Sparse-convolution export wrappers are inference-only.")
        if input_tensor.benchmark:
            raise NotImplementedError("Sparse-convolution benchmark mode is not supported.")
        if self.conv1x1:
            raise NotImplementedError(
                "Sparse-convolution export wrappers do not support conv1x1 fast paths."
            )

    def _resolve_algorithm(self, input_tensor: SparseConvTensor) -> ConvAlgo:
        """Resolve the convolution algorithm, validating indice-key reuse when needed."""
        algo = self.algo
        if self.indice_key is not None:
            cached_data = input_tensor.find_indice_pair(self.indice_key)
            if cached_data is not None and algo != cached_data.algo:
                raise ValueError(
                    "Sparse-convolution layers sharing an indice key must use the same algorithm."
                )
        return algo

    def _get_profile_context(self, input_tensor: SparseConvTensor, sparse_unique_name: str):
        """Return the profiling context used by spconv timers."""
        if input_tensor._timer is None or not sparse_unique_name:
            return nullcontext()
        return input_tensor._timer.namespace(sparse_unique_name)

    def _get_generation_profile_context(self, input_tensor: SparseConvTensor):
        """Return the timer namespace used while generating sparse indices."""
        if input_tensor._timer is None:
            return nullcontext()
        return input_tensor._timer.namespace("gen_pairs")

    def _store_indice_data(self, indice_dict: dict[str, Any], indice_data: Any) -> None:
        """Cache generated indice metadata under the configured indice key."""
        if self.indice_key is None:
            return
        msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
        assert self.indice_key not in indice_dict, msg
        indice_dict[self.indice_key] = indice_data

    def _finalize_output_tensor(
        self,
        out_tensor: SparseConvTensor,
        *,
        out_features: torch.Tensor,
        outids: torch.Tensor,
        indice_dict: dict[str, Any],
        out_spatial_shape: list[int],
        add_input: SparseConvTensor | None,
    ) -> SparseConvTensor:
        """Populate the exported sparse output tensor with computed metadata."""
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        if add_input is not None:
            out_tensor = out_tensor.replace_feature(
                _apply_activation(
                    out_tensor.features + add_input.features,
                    self.act_type,
                    self.act_alpha,
                    self.act_beta,
                )
            )
        return out_tensor

    def _resolve_native_plan(
        self,
        *,
        input_tensor: SparseConvTensor,
        indices: torch.Tensor,
        spatial_shape: list[int],
        batch_size: int,
        out_spatial_shape: list[int],
        indice_dict: dict[str, Any],
        algo: ConvAlgo,
    ) -> _NativeExecutionPlan:
        """Resolve sparse metadata for the native convolution path."""
        data = input_tensor.find_indice_pair(self.indice_key)
        if data is not None:
            assert isinstance(data, IndiceData)

        if self.inverse:
            assert data is not None and self.indice_key is not None
            assert data.is_subm is False, (
                "inverse conv can only be used with standard conv and pool ops."
            )
            self._check_inverse_reuse_valid(input_tensor, spatial_shape, data)
            return _NativeExecutionPlan(
                outids=data.indices,
                indice_pairs=data.indice_pairs,
                indice_pair_num=data.indice_pair_num,
                num_act_out=data.indices.shape[0],
                out_spatial_shape=data.spatial_shape,
            )

        if self.indice_key is not None and data is not None:
            assert self.subm, "only support reuse subm indices"
            self._check_subm_reuse_valid(input_tensor, spatial_shape, data)
            return _NativeExecutionPlan(
                outids=data.out_indices,
                indice_pairs=data.indice_pairs,
                indice_pair_num=data.indice_pair_num,
                num_act_out=data.out_indices.shape[0],
                out_spatial_shape=out_spatial_shape,
            )

        try:
            outids, indice_pairs, indice_pair_num, num_act_out = Fsp_custom.get_indice_pairs(
                indices,
                batch_size,
                spatial_shape,
                algo,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
                self.subm,
                self.transposed,
            )
        except Exception as exc:
            spconv_save_debug_data(indices)
            _raise_generation_error(
                exc,
                kind="native_pair",
                indices=tuple(indices.shape),
                bs=batch_size,
                ss=spatial_shape,
                algo=algo,
                ksize=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                subm=self.subm,
                transpose=self.transposed,
            )

        self._store_indice_data(
            indice_dict,
            IndiceData(
                outids,
                indices,
                indice_pairs,
                indice_pair_num,
                spatial_shape,
                out_spatial_shape,
                is_subm=self.subm,
                algo=algo,
                ksize=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
        )
        return _NativeExecutionPlan(
            outids=outids,
            indice_pairs=indice_pairs,
            indice_pair_num=indice_pair_num,
            num_act_out=num_act_out,
            out_spatial_shape=out_spatial_shape,
        )

    def _resolve_implicit_gemm_plan(
        self,
        *,
        input_tensor: SparseConvTensor,
        indices: torch.Tensor,
        spatial_shape: list[int],
        batch_size: int,
        out_spatial_shape: list[int],
        indice_dict: dict[str, Any],
        algo: ConvAlgo,
    ) -> _ImplicitGemmExecutionPlan:
        """Resolve sparse metadata for the implicit-GEMM execution path."""
        data = input_tensor.find_indice_pair(self.indice_key)
        if data is not None:
            assert isinstance(data, ImplicitGemmIndiceData)

        if self.inverse:
            assert data is not None and self.indice_key is not None
            assert data.is_subm is False, (
                "inverse conv can only be used with standard conv and pool ops."
            )
            self._check_inverse_reuse_valid(input_tensor, spatial_shape, data)
            return _ImplicitGemmExecutionPlan(
                outids=data.indices,
                pair_fwd=data.pair_bwd,
                pair_mask_fwd_splits=data.pair_mask_bwd_splits,
                mask_argsort_fwd_splits=data.mask_argsort_bwd_splits,
                masks=data.masks,
                num_act_out=data.indices.shape[0],
                out_spatial_shape=data.spatial_shape,
            )

        if self.indice_key is not None and data is not None:
            assert self.subm, "only support reuse subm indices"
            self._check_subm_reuse_valid(input_tensor, spatial_shape, data)
            return _ImplicitGemmExecutionPlan(
                outids=data.out_indices,
                pair_fwd=data.pair_fwd,
                pair_mask_fwd_splits=data.pair_mask_fwd_splits,
                mask_argsort_fwd_splits=data.mask_argsort_fwd_splits,
                masks=data.masks,
                num_act_out=data.out_voxel_num,
                out_spatial_shape=out_spatial_shape,
            )

        with self._get_generation_profile_context(input_tensor):
            try:
                outids, pair_fwd, pair_mask_fwd_splits, mask_argsort_fwd_splits, num_act_out = (
                    Fsp_custom.get_indice_pairs_implicit_gemm(
                        indices,
                        batch_size,
                        spatial_shape,
                        algo,
                        self.kernel_size,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.output_padding,
                        self.subm,
                        self.transposed,
                        not self.subm,
                        input_tensor.thrust_allocator,
                        input_tensor._timer,
                    )
                )
            except Exception as exc:
                spconv_save_debug_data(indices)
                _raise_generation_error(
                    exc,
                    kind="implicit_gemm_pair",
                    indices=tuple(indices.shape),
                    bs=batch_size,
                    ss=spatial_shape,
                    algo=algo,
                    ksize=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    subm=self.subm,
                    transpose=self.transposed,
                )

        masks = [None]
        self._store_indice_data(
            indice_dict,
            ImplicitGemmIndiceData(
                outids,
                indices,
                pair_fwd,
                None,
                pair_mask_fwd_splits,
                None,
                mask_argsort_fwd_splits,
                mask_argsort_bwd_splits=None,
                masks=masks,
                is_subm=self.subm,
                spatial_shape=spatial_shape,
                out_spatial_shape=out_spatial_shape,
                algo=algo,
                ksize=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                out_voxel_num=num_act_out,
            ),
        )
        return _ImplicitGemmExecutionPlan(
            outids=outids,
            pair_fwd=pair_fwd,
            pair_mask_fwd_splits=pair_mask_fwd_splits,
            mask_argsort_fwd_splits=mask_argsort_fwd_splits,
            masks=masks,
            num_act_out=num_act_out,
            out_spatial_shape=out_spatial_shape,
        )

    def _run_native_convolution(
        self,
        *,
        features: torch.Tensor,
        weight: torch.Tensor,
        plan: _NativeExecutionPlan,
        algo: ConvAlgo,
        timer: Any,
        bias: torch.Tensor | None,
        act_alpha: float,
        act_beta: float,
        act_type: tv.gemm.Activation,
    ) -> torch.Tensor:
        """Execute the native sparse-convolution branch."""
        assert not self.inverse, "Native inverse sparse-convolution export is not supported."
        indice_pairs = plan.indice_pairs
        if indice_pairs.device != features.device:
            indice_pairs = indice_pairs.to(features.device)
        return Fsp_custom.indice_conv(
            features,
            weight,
            indice_pairs,
            plan.indice_pair_num,
            plan.num_act_out,
            algo,
            False,
            self.subm,
            timer,
            bias,
            act_alpha,
            act_beta,
            act_type,
        )

    def _run_implicit_gemm_convolution(
        self,
        *,
        features: torch.Tensor,
        weight: torch.Tensor,
        plan: _ImplicitGemmExecutionPlan,
        timer: Any,
        bias: torch.Tensor | None,
        channel_scale: torch.Tensor | None,
        output_scale: float | None,
        add_input: SparseConvTensor | None,
        act_alpha: float,
        act_beta: float,
        act_type: tv.gemm.Activation,
    ) -> torch.Tensor:
        """Execute the implicit-GEMM sparse-convolution branch."""
        output_dtype = None if output_scale is not None else weight.dtype
        out_features = Fsp_custom.implicit_gemm(
            features,
            weight,
            plan.pair_fwd,
            plan.pair_mask_fwd_splits,
            plan.mask_argsort_fwd_splits,
            plan.num_act_out,
            plan.masks,
            False,
            self.subm,
            timer,
            self.fp32_accum,
            None,
            act_alpha,
            act_beta,
            act_type,
            1.0 if output_scale is None else output_scale,
            channel_scale,
            add_input.features if add_input is not None else None,
            0.0,
            output_dtype,
        )
        if bias is not None:
            out_features = out_features + bias
        return out_features

    def _conv_forward(
        self,
        training: bool,
        input: SparseConvTensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        add_input: SparseConvTensor | None = None,
        channel_scale: torch.Tensor | None = None,
        output_scale: float | None = None,
        name: str | None = None,
        sparse_unique_name: str = "",
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
        act_alpha: float = 0,
        act_beta: float = 0,
    ):
        """Run sparse convolution in inference mode.

        Args:
            training: Whether the wrapper is called in training mode.
            input: Sparse convolution input tensor.
            weight: Sparse convolution weights.
            bias: Optional bias tensor.
            add_input: Optional tensor added to the output.
            channel_scale: Optional per-channel scaling tensor.
            output_scale: Optional scalar output scale.
            name: Optional compatibility argument kept for the base spconv API.
            sparse_unique_name: Optional unique name used for profiling.
            act_type: Activation type.
            act_alpha: Activation alpha parameter.
            act_beta: Activation beta parameter.

        Returns:
            Output sparse convolution tensor.
        """
        del name
        self._validate_export_configuration(input, weight, training=training)
        if input.features.shape[1] != self.in_channels:
            raise ValueError(
                "Sparse-convolution input channel mismatch: "
                f"expected {self.in_channels}, got {input.features.shape[1]}."
            )
        features = (
            input.features.contiguous() if not input.features.is_contiguous() else input.features
        )
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        out_spatial_shape = _resolve_output_spatial_shape(
            spatial_shape,
            subm=self.subm,
            transposed=self.transposed,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
        )
        out_tensor = input.shadow_copy()

        indice_dict = input.indice_dict.copy()
        algo = self._resolve_algorithm(input)
        profile_ctx = self._get_profile_context(input, sparse_unique_name)
        with profile_ctx:
            if algo == ConvAlgo.Native:
                native_plan = self._resolve_native_plan(
                    input_tensor=input,
                    indices=indices,
                    spatial_shape=spatial_shape,
                    batch_size=batch_size,
                    out_spatial_shape=out_spatial_shape,
                    indice_dict=indice_dict,
                    algo=algo,
                )
                outids = native_plan.outids
                out_spatial_shape = native_plan.out_spatial_shape
                out_features = self._run_native_convolution(
                    features=features,
                    weight=weight,
                    plan=native_plan,
                    algo=algo,
                    timer=input._timer,
                    bias=bias,
                    act_alpha=act_alpha,
                    act_beta=act_beta,
                    act_type=act_type,
                )
            else:
                implicit_plan = self._resolve_implicit_gemm_plan(
                    input_tensor=input,
                    indices=indices,
                    spatial_shape=spatial_shape,
                    batch_size=batch_size,
                    out_spatial_shape=out_spatial_shape,
                    indice_dict=indice_dict,
                    algo=algo,
                )
                outids = implicit_plan.outids
                out_spatial_shape = implicit_plan.out_spatial_shape
                out_features = self._run_implicit_gemm_convolution(
                    features=features,
                    weight=weight,
                    plan=implicit_plan,
                    timer=input._timer,
                    bias=bias,
                    channel_scale=channel_scale,
                    output_scale=output_scale,
                    add_input=add_input,
                    act_alpha=act_alpha,
                    act_beta=act_beta,
                    act_type=act_type,
                )

        if not self.subm and not self.inverse and self.record_voxel_count:
            if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
                ops.maximum_value_int_(
                    getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING), outids.shape[0]
                )
        return self._finalize_output_tensor(
            out_tensor,
            out_features=out_features,
            outids=outids,
            indice_dict=indice_dict,
            out_spatial_shape=out_spatial_shape,
            add_input=add_input,
        )


class SparseConv3d(SparseConvolution):
    """Implement a 3D sparse convolution backed by the exportable wrapper.

    This layer mirrors ``spconv.SparseConv3d`` while keeping export-compatible
    execution behavior.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        """Initialize the sparse 3D convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Sparse convolution kernel size.
            stride: Sparse convolution stride.
            padding: Sparse convolution padding.
            dilation: Sparse convolution dilation.
            groups: Number of convolution groups.
            bias: Whether to include a learnable bias.
            indice_key: Optional spconv indice cache key.
            algo: Sparse convolution algorithm.
            fp32_accum: Whether to accumulate in FP32.
            name: Optional debug or profiling name.
        """
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SubMConv3d(SparseConvolution):
    """Implement a 3D submanifold sparse convolution with export support.

    This layer mirrors ``spconv.SubMConv3d`` while preserving deployment-aware
    execution paths.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        """Initialize the submanifold sparse convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Sparse convolution kernel size.
            stride: Sparse convolution stride.
            padding: Sparse convolution padding.
            dilation: Sparse convolution dilation.
            groups: Number of convolution groups.
            bias: Whether to include a learnable bias.
            indice_key: Optional spconv indice cache key.
            algo: Sparse convolution algorithm.
            fp32_accum: Whether to accumulate in FP32.
            name: Optional debug or profiling name.
        """
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )
