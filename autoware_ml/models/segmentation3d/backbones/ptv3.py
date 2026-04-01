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

"""Reusable PTv3 backbone components.

This module contains the reusable encoder-decoder blocks used by PTv3.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import math
from collections.abc import Sequence
from typing import Any

import spconv.pytorch as spconv
import torch
import torch.nn as nn

from autoware_ml.ops.indexing import argsort, unique
from autoware_ml.ops.segment import segment_csr
from autoware_ml.utils.point_cloud import Point, offset_to_bincount
from autoware_ml.ops.spconv import SubMConv3d as ExportableSubMConv3d


def load_flash_attn_module() -> Any:
    """Import the ``flash_attn`` package used by serialized attention blocks.

    Returns:
        Imported ``flash_attn`` module.

    Raises:
        ImportError: Raised when ``flash_attn`` is not installed.
    """
    return importlib.import_module("flash_attn")


def is_sparse_conv_module(module: nn.Module) -> bool:
    """Return whether a module consumes sparse convolution tensors.

    Args:
        module: Module inspected for sparse-convolution semantics.

    Returns:
        ``True`` when the module expects sparse convolution tensors.
    """
    return spconv.modules.is_spconv_module(module) or isinstance(module, ExportableSubMConv3d)


def replace_submconv3d_for_export(module: nn.Module) -> None:
    """Replace native ``spconv.SubMConv3d`` layers with exportable wrappers.

    Args:
        module: Module hierarchy traversed in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, ExportableSubMConv3d):
            continue
        if isinstance(child, spconv.SubMConv3d):
            exportable_child = ExportableSubMConv3d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                indice_key=child.indice_key,
                algo=child.algo,
                fp32_accum=child.fp32_accum,
                name=getattr(child, "name", None),
            )
            exportable_child = exportable_child.to(
                device=child.weight.device, dtype=child.weight.dtype
            )
            exportable_child.load_state_dict(child.state_dict())
            module._modules[name] = exportable_child
            continue
        replace_submconv3d_for_export(child)


class PointModule(nn.Module):
    """Base class for modules that operate on :class:`Point` containers.

    Subclasses consume and return :class:`Point` objects, allowing PTv3 blocks
    to compose sparse, dense, and point-wise operations behind one interface.
    """


class PointSequential(PointModule):
    """Compose modules that operate on point, sparse, or dense tensors.

    The container dispatches each child according to whether it expects a
    :class:`Point`, a sparse-convolution tensor, or a dense tensor.
    """

    def __init__(self, *modules: nn.Module) -> None:
        """Initialize the sequential point-module container.

        Args:
            *modules: Modules appended to the sequential container.
        """
        super().__init__()
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def add(self, module: nn.Module, name: str | None = None) -> None:
        """Append a module to the sequence.

        Args:
            module: Module to append.
            name: Optional module name used inside the container.
        """
        self.add_module(name or str(len(self._modules)), module)

    def forward(self, input_data: Point | spconv.SparseConvTensor | torch.Tensor) -> Any:
        """Apply the contained modules to point, sparse, or dense inputs.

        Args:
            input_data: Point container, sparse tensor, or dense tensor.

        Returns:
            Output produced by the sequential module stack.
        """
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input_data = module(input_data)
            elif is_sparse_conv_module(module):
                if isinstance(input_data, Point):
                    input_data.sparse_conv_feat = module(input_data.sparse_conv_feat)
                    input_data.feat = input_data.sparse_conv_feat.features
                else:
                    input_data = module(input_data)
            else:
                if isinstance(input_data, Point):
                    input_data.feat = module(input_data.feat)
                    if "sparse_conv_feat" in input_data:
                        input_data.sparse_conv_feat = input_data.sparse_conv_feat.replace_feature(
                            input_data.feat
                        )
                elif isinstance(input_data, spconv.SparseConvTensor):
                    input_data = input_data.replace_feature(module(input_data.features))
                else:
                    input_data = module(input_data)
        return input_data


class DropPath(nn.Module):
    """Apply stochastic depth to residual branches during training.

    This layer randomly drops the residual path for whole samples while keeping
    the expected activation magnitude unchanged.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize the stochastic depth layer.

        Args:
            drop_prob: Probability of dropping the residual branch.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth during training.

        Args:
            x: Input tensor.

        Returns:
            Tensor after stochastic-depth masking.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor.div_(keep_prob)


class RelativePositionEncoding(nn.Module):
    """Encode relative point offsets into attention biases.

    The table is indexed by clipped relative coordinates and summed across
    spatial axes to produce per-head attention bias values.
    """

    def __init__(self, patch_size: int, num_heads: int) -> None:
        """Initialize the relative position encoding table.

        Args:
            patch_size: Window size used by serialized attention.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.patch_size = patch_size
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        """Encode relative coordinates into attention biases.

        Args:
            coord: Relative coordinate tensor.

        Returns:
            Relative attention bias tensor.
        """
        index = coord.clamp(-self.pos_bnd, self.pos_bnd) + self.pos_bnd
        index = index + torch.arange(3, device=coord.device) * self.rpe_num
        output = self.table.index_select(0, index.reshape(-1))
        output = output.view(index.shape + (-1,)).sum(3)
        return output.permute(0, 3, 1, 2)


class SerializedAttention(PointModule):
    """Apply windowed self-attention over serialized point tokens.

    PTv3 groups points according to a serialization order and then performs
    local self-attention inside fixed-size windows over that ordering.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        order_index: int,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
    ) -> None:
        """Initialize serialized attention.

        Args:
            channels: Input and output feature dimension.
            num_heads: Number of attention heads.
            patch_size: Maximum serialized window size.
            qkv_bias: Whether to use learnable bias in the QKV projection.
            qk_scale: Optional manual scale for query-key attention.
            attn_drop: Dropout applied to attention weights.
            proj_drop: Dropout applied after the output projection.
            order_index: Serialization order index consumed by this block.
            enable_rpe: Whether to use relative positional encoding.
            enable_flash: Whether to use flash attention.
            upcast_attention: Whether to upcast Q/K before attention.
            upcast_softmax: Whether to upcast logits before softmax.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if enable_flash and enable_rpe:
            raise ValueError(
                "Flash attention does not support relative positional encoding in PTv3."
            )
        if enable_flash and upcast_attention:
            raise ValueError("Flash attention requires upcast_attention=false in PTv3.")
        if enable_flash and upcast_softmax:
            raise ValueError("Flash attention requires upcast_softmax=false in PTv3.")

        self.channels = channels
        self.num_heads = num_heads
        self.enable_flash = enable_flash
        self.patch_size_max = patch_size
        self.patch_size = patch_size if enable_flash else 0
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.export_mode = False
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RelativePositionEncoding(patch_size, num_heads) if enable_rpe else None
        self.flash_attn = None

    @torch.no_grad()
    def _get_padding_and_inverse(
        self, point: Point
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute padded token indices and inverse mapping for windowed attention.

        Args:
            point: Serialized point container.

        Returns:
            Tuple containing padded ordering indices, inverse indices, and
            optional cumulative sequence lengths for flash attention.
        """
        bincount = offset_to_bincount(point.offset)
        padded_bincount = (
            torch.maximum(
                torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc"),
                torch.ones_like(bincount),
            )
            * self.patch_size
        )
        mask = bincount > self.patch_size
        padded_bincount = (~mask).long() * bincount + mask.long() * padded_bincount

        if self.export_mode:
            if point.offset.numel() != 1:
                raise ValueError("PTv3 export mode supports only single-sample export batches.")
            pad = torch.arange(padded_bincount[0], device=point.offset.device)
            unpad = torch.arange(point.offset[0], device=point.offset.device)
            if bincount[0] != padded_bincount[0]:
                pad[
                    padded_bincount[0]
                    - self.patch_size
                    + (bincount[0] % self.patch_size) : padded_bincount[0]
                ] = pad[
                    padded_bincount[0]
                    - 2 * self.patch_size
                    + (bincount[0] % self.patch_size) : padded_bincount[0] - self.patch_size
                ]
            if not self.enable_flash:
                return pad, unpad, None
            cu_seqlens = torch.arange(
                0,
                padded_bincount[0],
                step=self.patch_size,
                dtype=torch.int32,
                device=point.offset.device,
            )
            return pad, unpad, nn.functional.pad(cu_seqlens, (0, 1), value=padded_bincount[0])

        offset = nn.functional.pad(point.offset, (1, 0))
        padded_offset = nn.functional.pad(torch.cumsum(padded_bincount, dim=0), (1, 0))
        pad = torch.arange(padded_offset[-1], device=point.offset.device)
        unpad = torch.arange(offset[-1], device=point.offset.device)
        cu_seqlens = [] if self.enable_flash else None
        for batch_index in range(point.offset.numel()):
            unpad[offset[batch_index] : offset[batch_index + 1]] += (
                padded_offset[batch_index] - offset[batch_index]
            )
            if bincount[batch_index] != padded_bincount[batch_index]:
                pad[
                    padded_offset[batch_index + 1]
                    - self.patch_size
                    + (bincount[batch_index] % self.patch_size) : padded_offset[batch_index + 1]
                ] = pad[
                    padded_offset[batch_index + 1]
                    - 2 * self.patch_size
                    + (bincount[batch_index] % self.patch_size) : padded_offset[batch_index + 1]
                    - self.patch_size
                ]
            pad[padded_offset[batch_index] : padded_offset[batch_index + 1]] -= (
                padded_offset[batch_index] - offset[batch_index]
            )
            if cu_seqlens is not None:
                cu_seqlens.append(
                    torch.arange(
                        padded_offset[batch_index],
                        padded_offset[batch_index + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=point.offset.device,
                    )
                )
        if cu_seqlens is None:
            return pad, unpad, None
        return pad, unpad, nn.functional.pad(torch.cat(cu_seqlens), (0, 1), value=padded_offset[-1])

    @torch.no_grad()
    def disable_flash(self) -> None:
        """Disable flash attention for export-oriented execution paths."""
        self.enable_flash = False
        self.patch_size = 0
        self.flash_attn = None

    def forward(self, point: Point) -> Point:
        """Apply serialized self-attention to point features.

        Args:
            point: Point container with serialized ordering metadata.

        Returns:
            Point container with updated features.
        """
        head_count = self.num_heads
        if not self.enable_flash:
            min_points = int(offset_to_bincount(point.offset).min().item())
            self.patch_size = max(1, min(self.patch_size_max, min_points))
        patch_size = self.patch_size
        channel_count = self.channels
        pad, unpad, cu_seqlens = self._get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            qkv = qkv.reshape(-1, patch_size, 3, head_count, channel_count // head_count)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attention = (q * self.scale) @ k.transpose(-2, -1)
            if self.rpe is not None:
                grid_coord = point.grid_coord[order].reshape(-1, patch_size, 3)
                rel_pos = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
                attention = attention + self.rpe(rel_pos)
            if self.upcast_softmax:
                attention = attention.float()
            attention = self.softmax(attention)
            attention = self.attn_drop(attention).to(qkv.dtype)
            feat = (attention @ v).transpose(1, 2).reshape(-1, channel_count)
        else:
            assert cu_seqlens is not None
            if self.flash_attn is None:
                self.flash_attn = load_flash_attn_module()
            feat = self.flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, head_count, channel_count // head_count),
                cu_seqlens,
                max_seqlen=patch_size,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, channel_count)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]
        point.feat = self.proj_drop(self.proj(feat))
        return point


class MLP(nn.Module):
    """Apply the feed-forward sublayer used inside each PTv3 block.

    The module expands the channel dimension, applies GELU activation, and
    projects features back to the original dimension with dropout.
    """

    def __init__(self, channels: int, mlp_ratio: float, drop: float) -> None:
        """Initialize the feed-forward network.

        Args:
            channels: Input and output feature dimension.
            mlp_ratio: Hidden-layer expansion ratio.
            drop: Dropout applied between linear layers.
        """
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Input tensor.

        Returns:
            Tensor after the MLP stack.
        """
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(PointModule):
    """Compose positional encoding, attention, and MLP for one PTv3 block.

    Each block first injects sparse-convolutional positional information and
    then applies serialized attention and an MLP with residual connections.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        pre_norm: bool,
        order_index: int,
        cpe_indice_key: str,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
    ) -> None:
        """Initialize one PTv3 attention block.

        Args:
            channels: Input and output feature dimension.
            num_heads: Number of attention heads.
            patch_size: Maximum serialized window size.
            mlp_ratio: Hidden-layer expansion ratio for the MLP.
            qkv_bias: Whether to use learnable bias in the QKV projection.
            qk_scale: Optional manual scale for query-key attention.
            attn_drop: Dropout applied to attention weights.
            proj_drop: Dropout applied after output projections.
            drop_path: Stochastic-depth probability.
            pre_norm: Whether to apply pre-normalization.
            order_index: Serialization order index consumed by this block.
            cpe_indice_key: Sparse convolution indice key for positional encoding.
            enable_rpe: Whether to use relative positional encoding.
            enable_flash: Whether to use flash attention.
            upcast_attention: Whether to upcast Q/K before attention.
            upcast_softmax: Whether to upcast logits before softmax.
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key
            ),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.norm1 = PointSequential(nn.LayerNorm(channels))
        self.attn = SerializedAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(nn.LayerNorm(channels))
        self.mlp = PointSequential(MLP(channels, mlp_ratio, proj_drop))
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0 else nn.Identity())

    def forward(self, point: Point) -> Point:
        """Apply convolutional positional encoding, attention, and MLP.

        Args:
            point: Point container processed by the block.

        Returns:
            Point container with updated features.
        """
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    """Pool serialized point groups into a coarser hierarchy level.

    The pooling stage aggregates features and coordinates per serialized group
    and records the inverse mapping required by the decoder.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, shuffle_orders: bool
    ) -> None:
        """Initialize serialized pooling.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            stride: Hierarchical stride factor.
            shuffle_orders: Whether to shuffle serialization orders after pooling.
        """
        super().__init__()
        self.stride = stride
        self.shuffle_orders = shuffle_orders
        self.export_mode = False
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))
        self.act = PointSequential(nn.GELU())

    def forward(self, point: Point) -> Point:
        """Pool serialized point features into a coarser hierarchy level.

        Args:
            point: Point container at the current hierarchy level.

        Returns:
            Coarser point container with pooled features.
        """
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > int(point.serialized_depth.item()):
            pooling_depth = 0

        code = point.serialized_code >> (pooling_depth * 3)
        if self.export_mode:
            pooled_code0, cluster, counts, _ = unique(code[0])
            indices = argsort(cluster)
        else:
            pooled_code0, cluster, counts = torch.unique(
                code[0], sorted=True, return_inverse=True, return_counts=True
            )
            indices = torch.argsort(cluster)
        del pooled_code0
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        pooled_code = code[:, head_indices]
        if self.export_mode:
            pooled_order = torch.stack(
                [argsort(pooled_code[row_index]) for row_index in range(pooled_code.shape[0])],
                dim=0,
            )
        else:
            pooled_order = torch.argsort(pooled_code, dim=1)
        pooled_inverse = torch.zeros_like(pooled_order).scatter_(
            1,
            pooled_order,
            torch.arange(pooled_code.shape[1], device=pooled_order.device).repeat(
                pooled_code.shape[0], 1
            ),
        )
        if self.shuffle_orders:
            permutation = torch.randperm(pooled_code.shape[0], device=pooled_code.device)
            pooled_code = pooled_code[permutation]
            pooled_order = pooled_order[permutation]
            pooled_inverse = pooled_inverse[permutation]

        scatter_feat = segment_csr(self.proj(point.feat)[indices], idx_ptr, "max")
        scatter_coord = segment_csr(point.coord[indices], idx_ptr, "mean")
        pooled = Point(
            feat=scatter_feat,
            coord=scatter_coord,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=pooled_code,
            serialized_order=pooled_order,
            serialized_inverse=pooled_inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            sparse_shape=point.sparse_shape >> pooling_depth,
            pooling_inverse=cluster,
            pooling_parent=point,
            offset=(
                point.batch.new_tensor([head_indices.numel()], dtype=torch.long)
                if point.batch[head_indices].numel() == 0
                or point.batch[head_indices][-1].item() == 0
                else torch.cumsum(point.batch[head_indices].bincount(), dim=0).long()
            ),
        )
        pooled = self.norm(pooled)
        pooled = self.act(pooled)
        pooled.sparsify()
        return pooled


class SerializedUnpooling(PointModule):
    """Restore one hierarchy level and fuse skip features in the decoder.

    The decoder projects coarse features, projects skip features from the
    parent level, and merges them through the saved pooling inverse.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialize serialized unpooling.

        Args:
            in_channels: Decoder feature dimension.
            skip_channels: Skip-connection feature dimension.
            out_channels: Output feature dimension after fusion.
        """
        super().__init__()
        self.proj = PointSequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )
        self.proj_skip = PointSequential(
            nn.Linear(skip_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )

    def forward(self, point: Point) -> Point:
        """Unpool one hierarchy level and fuse skip features.

        Args:
            point: Point container at the coarse hierarchy level.

        Returns:
            Point container fused with skip features.
        """
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        return parent


class Embedding(PointModule):
    """Embed raw point features with the PTv3 sparse-convolution stem.

    This stem converts input features into the first encoder channel width
    before the hierarchical PTv3 stages are applied.
    """

    def __init__(self, in_channels: int, embed_channels: int) -> None:
        """Initialize the sparse-convolution embedding stem.

        Args:
            in_channels: Input feature dimension.
            embed_channels: Output embedding dimension.
        """
        super().__init__()
        self.stem = PointSequential(
            spconv.SubMConv3d(
                in_channels, embed_channels, kernel_size=5, padding=1, bias=False, indice_key="stem"
            ),
            nn.BatchNorm1d(embed_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )

    def forward(self, point: Point) -> Point:
        """Embed raw point features with a sparse convolution stem.

        Args:
            point: Point container with raw per-point features.

        Returns:
            Point container with embedded features.
        """
        return self.stem(point)


class PointTransformerV3Backbone(PointModule):
    """Implement the PTv3 encoder-decoder backbone for point segmentation.

    The backbone serializes points, applies hierarchical encoder blocks with
    sparse pooling, and reconstructs fine-grained features through decoder
    stages with skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        order: Sequence[str],
        stride: Sequence[int],
        enc_depths: Sequence[int],
        enc_channels: Sequence[int],
        enc_num_head: Sequence[int],
        enc_patch_size: Sequence[int],
        dec_depths: Sequence[int],
        dec_channels: Sequence[int],
        dec_num_head: Sequence[int],
        dec_patch_size: Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        pre_norm: bool,
        shuffle_orders: bool,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        cls_mode: bool,
        pdnorm_bn: bool,
        pdnorm_ln: bool,
        pdnorm_decouple: bool,
        pdnorm_adaptive: bool,
        pdnorm_affine: bool,
        pdnorm_conditions: Sequence[str],
    ) -> None:
        """Initialize the PTv3 encoder-decoder backbone.

        Args:
            in_channels: Input feature dimension.
            order: Serialization orders used by the backbone.
            stride: Pooling strides between encoder stages.
            enc_depths: Number of blocks per encoder stage.
            enc_channels: Encoder channel widths per stage.
            enc_num_head: Attention head counts per encoder stage.
            enc_patch_size: Attention patch sizes per encoder stage.
            dec_depths: Number of blocks per decoder stage.
            dec_channels: Decoder channel widths per stage.
            dec_num_head: Attention head counts per decoder stage.
            dec_patch_size: Attention patch sizes per decoder stage.
            mlp_ratio: Hidden-layer expansion ratio for each block MLP.
            qkv_bias: Whether to use learnable bias in QKV projections.
            qk_scale: Optional manual attention scale.
            attn_drop: Dropout applied to attention weights.
            proj_drop: Dropout applied after output projections.
            drop_path: Stochastic-depth probability.
            pre_norm: Whether to apply pre-normalization.
            shuffle_orders: Whether to shuffle serialization orders.
            enable_rpe: Whether to use relative positional encoding.
            enable_flash: Whether to use flash attention.
            upcast_attention: Whether to upcast Q/K before attention.
            upcast_softmax: Whether to upcast logits before softmax.
            cls_mode: Whether to run in classification mode.
            pdnorm_bn: Whether to enable PDNorm batch normalization.
            pdnorm_ln: Whether to enable PDNorm layer normalization.
            pdnorm_decouple: Whether to decouple PDNorm statistics.
            pdnorm_adaptive: Whether to use adaptive PDNorm.
            pdnorm_affine: Whether to use affine PDNorm parameters.
            pdnorm_conditions: PDNorm condition identifiers.
        """
        super().__init__()
        if any([pdnorm_bn, pdnorm_ln, pdnorm_decouple, pdnorm_adaptive is True]) and (
            pdnorm_bn or pdnorm_ln
        ):
            raise ValueError("PDNorm is not integrated in autoware-ml PTv3 yet.")
        if cls_mode:
            raise ValueError("Classification mode is not supported by the PTv3 port.")
        del pdnorm_affine, pdnorm_conditions

        self.order = list(order)
        self.shuffle_orders = shuffle_orders
        stage_count = len(enc_depths)
        self.embedding = Embedding(in_channels, enc_channels[0])

        enc_drop_path = [value.item() for value in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for stage_index in range(stage_count):
            encoder = PointSequential()
            if stage_index > 0:
                encoder.add(
                    SerializedPooling(
                        enc_channels[stage_index - 1],
                        enc_channels[stage_index],
                        stride[stage_index - 1],
                        shuffle_orders,
                    ),
                    name="down",
                )
            for block_index in range(enc_depths[stage_index]):
                encoder.add(
                    Block(
                        channels=enc_channels[stage_index],
                        num_heads=enc_num_head[stage_index],
                        patch_size=enc_patch_size[stage_index],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path[sum(enc_depths[:stage_index]) + block_index],
                        pre_norm=pre_norm,
                        order_index=block_index % len(self.order),
                        cpe_indice_key=f"stage{stage_index}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{block_index}",
                )
            self.enc.add(encoder, name=f"enc{stage_index}")

        dec_drop_path = [value.item() for value in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        decoder_channels = list(dec_channels) + [enc_channels[-1]]
        for stage_index in reversed(range(stage_count - 1)):
            decoder = PointSequential()
            decoder.add(
                SerializedUnpooling(
                    decoder_channels[stage_index + 1],
                    enc_channels[stage_index],
                    decoder_channels[stage_index],
                ),
                name="up",
            )
            stage_drop = dec_drop_path[
                sum(dec_depths[:stage_index]) : sum(dec_depths[: stage_index + 1])
            ]
            stage_drop.reverse()
            for block_index in range(dec_depths[stage_index]):
                decoder.add(
                    Block(
                        channels=decoder_channels[stage_index],
                        num_heads=dec_num_head[stage_index],
                        patch_size=dec_patch_size[stage_index],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=stage_drop[block_index],
                        pre_norm=pre_norm,
                        order_index=block_index % len(self.order),
                        cpe_indice_key=f"stage{stage_index}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{block_index}",
                )
            self.dec.add(decoder, name=f"dec{stage_index}")

    def forward(self, data_dict: dict[str, torch.Tensor]) -> Point:
        """Serialize inputs, run the encoder-decoder, and return point features.

        Args:
            data_dict: Input tensors required by the PTv3 backbone.

        Returns:
            Point container with decoded point features.
        """
        point = Point(data_dict)
        point.serialization(self.order, self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point

    def set_serialization_order(self, order: Sequence[str]) -> None:
        """Update serialization order and reassign block order indices.

        This is mainly used by deployment code, where PTv3 export follows a
        narrower AWML-compatible serialization contract than training.

        Args:
            order: Serialization orders used by the backbone.
        """
        self.order = list(order)

        for stage in self.enc._modules.values():
            block_index = 0
            for module in stage._modules.values():
                if isinstance(module, Block):
                    module.attn.order_index = block_index % len(self.order)
                    block_index += 1

        for stage in self.dec._modules.values():
            block_index = 0
            for module in stage._modules.values():
                if isinstance(module, Block):
                    module.attn.order_index = block_index % len(self.order)
                    block_index += 1

    def prepare_export_copy(self, order: Sequence[str]) -> PointTransformerV3Backbone:
        """Return an isolated backbone copy configured for ONNX export.

        Args:
            order: Serialization orders used by the export graph.

        Returns:
            Export-ready PTv3 backbone copy.
        """
        export_backbone = deepcopy(self).eval()
        export_backbone.set_serialization_order(order)
        export_backbone.shuffle_orders = False
        replace_submconv3d_for_export(export_backbone)
        for module in export_backbone.modules():
            if isinstance(module, SerializedAttention):
                module.disable_flash()
                module.export_mode = True
            if isinstance(module, SerializedPooling):
                module.export_mode = True
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
        return export_backbone

    def export_forward(self, data_dict: dict[str, torch.Tensor]) -> Point:
        """Run the backbone with precomputed serialization metadata for export.

        Args:
            data_dict: Input tensors and precomputed serialization metadata.

        Returns:
            Point container with decoded point features.
        """
        point = Point(data_dict)
        point["serialized_depth"] = data_dict["serialized_depth"]
        point["serialized_code"] = data_dict["serialized_code"]
        point["serialized_order"] = data_dict["serialized_order"]
        point["serialized_inverse"] = data_dict["serialized_inverse"]
        point["sparse_shape"] = data_dict["sparse_shape"]
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point
