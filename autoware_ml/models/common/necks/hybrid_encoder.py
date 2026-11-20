# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
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

"""RT-DETR hybrid encoder, ported from RT-DETRv2's src/zoo/rtdetr/hybrid_encoder.py.

Attention-based intra-scale feature interaction (AIFI) on the coarsest level, followed by a
CNN-based cross-scale fusion (CCFF) top-down/bottom-up path built from CSPRep blocks.
Module names match the source ("v2" input projection), so its checkpoints load unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import copy

from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from autoware_ml.models.common.layers.conv import ConvModule, build_activation


def _conv_norm(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    act: str | None = None,
) -> ConvModule:
    """Build a convolution, batch norm and activation block as RT-DETR does.

    The batch norm keeps the PyTorch defaults that RT-DETR trains with, not the ones of
    :class:`ConvModule`, and the activation is off unless given.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        act: Activation name, or ``None`` for none.

    Returns:
        The block, with the ``conv`` and ``norm`` children of RT-DETR.
    """
    return ConvModule(
        in_channels,
        out_channels,
        stride,
        norm_eps=1e-5,
        norm_momentum=0.1,
        kernel_size=kernel_size,
        act=act,
    )


class RepVggBlock(nn.Module):
    """Parallel 3x3 and 1x1 conv-norm branches, fusable into one 3x3 conv for deployment.

    The block as trained, and after ``convert_to_deploy()`` folds both branches into ``conv``::

        training                               after convert_to_deploy()

        x ---------------+                     x
        |                |                     |
        conv1 3x3 + BN   conv2 1x1 + BN        conv 3x3 (with bias)
        |                |                     |
        +<---------------+                     act
        |                                      |
        act                                    out
        |
        out
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "silu") -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            act: Activation name applied to the summed branches.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = _conv_norm(in_channels, out_channels, 3, 1)
        self.conv2 = _conv_norm(in_channels, out_channels, 1, 1)
        self.conv: nn.Conv2d | None = None
        self.act = build_activation(act)

    def forward(
        self, x: Float[Tensor, "batch_size in_channels height width"]
    ) -> Float[Tensor, "batch_size out_channels height width"]:
        """Apply the block.

        Args:
            x: Input feature map.

        Returns:
            Output feature map, from the fused convolution once the block is converted.
        """
        if self.conv is not None:
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    @torch.no_grad()
    def convert_to_deploy(self) -> None:
        """Fold both branches and their batch norms into a single 3x3 conv named ``conv``."""
        kernel3x3, bias3x3 = _fuse_conv_norm(self.conv1)
        kernel1x1, bias1x1 = _fuse_conv_norm(self.conv2)
        if self.conv is None:
            self.conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                3,
                1,
                padding=1,
                device=kernel3x3.device,
                dtype=kernel3x3.dtype,
            )
        self.conv.weight.copy_(kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]))
        self.conv.bias.copy_(bias3x3 + bias1x1)


def _fuse_conv_norm(
    block: ConvModule,
) -> tuple[
    Float[Tensor, "out_channels in_channels kernel_height kernel_width"],
    Float[Tensor, " out_channels"],
]:
    """Return the kernel and bias of a bias-free conv followed by an eval-mode batch norm."""
    norm = block.norm
    std = (norm.running_var + norm.eps).sqrt()
    scale = norm.weight / std
    return block.conv.weight * scale.reshape(-1, 1, 1, 1), norm.bias - norm.running_mean * scale


class CSPRepLayer(nn.Module):
    """CSP block: a RepVgg-block path and a 1x1 bypass, summed and projected.

    Both paths start from the whole input, and ``hidden = out_channels * expansion``::

        x ----------------------------+
        |                             |
        conv1 1x1, hidden             conv2 1x1, hidden
        |                             |
        RepVggBlock x num_blocks      |
        |                             |
        +<----------------------------+
        |
        conv3 1x1, out_channels  (identity when hidden == out_channels)
        |
        out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        act: str = "silu",
    ) -> None:
        """Initialize the layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            num_blocks: Number of RepVgg blocks.
            expansion: Hidden channel count as a fraction of ``out_channels``.
            act: Activation name.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = _conv_norm(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = _conv_norm(in_channels, hidden_channels, 1, 1, act=act)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = _conv_norm(hidden_channels, out_channels, 1, 1, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(
        self, x: Float[Tensor, "batch_size in_channels height width"]
    ) -> Float[Tensor, "batch_size out_channels height width"]:
        """Apply the layer.

        Args:
            x: Input feature map.

        Returns:
            Output feature map.
        """
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class TransformerEncoderLayer(nn.Module):
    """Post-norm transformer encoder layer with a GELU feed-forward network."""

    def __init__(
        self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1
    ) -> None:
        """Initialize the layer.

        Args:
            d_model: Token dimension.
            num_heads: Number of attention heads.
            dim_feedforward: Hidden dimension of the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: Float[Tensor, "batch_size num_tokens d_model"],
        pos_embed: Float[Tensor, "1 num_tokens d_model"] | None = None,
    ) -> Float[Tensor, "batch_size num_tokens d_model"]:
        """Apply self-attention and the feed-forward network.

        Args:
            src: Tokens.
            pos_embed: Position embedding added to queries and keys, shared by the batch.

        Returns:
            Updated tokens.
        """
        q = k = src if pos_embed is None else src + pos_embed
        attended, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = self.norm1(src + self.dropout1(attended))
        ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(ffn))


class TransformerEncoder(nn.Module):
    """Stack of identical transformer encoder layers."""

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int) -> None:
        """Initialize the stack.

        Args:
            encoder_layer: Layer to replicate.
            num_layers: Number of layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        src: Float[Tensor, "batch_size num_tokens d_model"],
        pos_embed: Float[Tensor, "1 num_tokens d_model"] | None = None,
    ) -> Float[Tensor, "batch_size num_tokens d_model"]:
        """Apply every layer in turn.

        Args:
            src: Tokens.
            pos_embed: Position embedding, shared by the batch.

        Returns:
            Updated tokens.
        """
        for layer in self.layers:
            src = layer(src, pos_embed=pos_embed)
        return src


class HybridEncoder(nn.Module):
    """RT-DETR hybrid encoder: AIFI on selected levels, then CCFF fusion across levels."""

    def __init__(
        self,
        in_channels: Sequence[int] = (512, 1024, 2048),
        feat_strides: Sequence[int] = (8, 16, 32),
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        use_encoder_idx: Sequence[int] = (2,),
        num_encoder_layers: int = 1,
        pe_temperature: float = 10000.0,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        act: str = "silu",
        eval_spatial_size: Sequence[int] | None = None,
    ) -> None:
        """Initialize the encoder.

        Args:
            in_channels: Channel count of each input level.
            feat_strides: Stride of each input level.
            hidden_dim: Channel count of every output level.
            num_heads: Attention heads of the AIFI transformer.
            dim_feedforward: Feed-forward dimension of the AIFI transformer.
            dropout: Dropout of the AIFI transformer.
            use_encoder_idx: Input levels that go through AIFI, strictly increasing; empty for
                none.
            num_encoder_layers: Transformer layers per AIFI level.
            pe_temperature: Temperature of the 2D sin-cos position embedding.
            expansion: Hidden width of the CSPRep blocks as a fraction of ``hidden_dim``.
            depth_mult: Scales the number of RepVgg blocks per CSPRep block (3 at 1.0).
            act: Activation name of the CCFF convolutions.
            eval_spatial_size: Input (height, width) whose position embeddings are cached
                for inference. Other sizes, and training, compute them on the fly.

        Raises:
            ValueError: If ``use_encoder_idx`` is out of range or not strictly increasing.
        """
        super().__init__()
        num_levels = len(in_channels)
        encoder_idx = tuple(int(i) for i in use_encoder_idx)
        if any(i < 0 or i >= num_levels for i in encoder_idx):
            raise ValueError(
                f"use_encoder_idx must select levels in [0, {num_levels - 1}], got {encoder_idx}."
            )
        if any(later <= earlier for earlier, later in zip(encoder_idx, encoder_idx[1:])):
            raise ValueError(f"use_encoder_idx must be strictly increasing, got {encoder_idx}.")
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.out_channels = [hidden_dim] * len(self.in_channels)
        self.out_strides = self.feat_strides

        self.input_proj = nn.ModuleList(
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )
            for channels in self.in_channels
        )

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.encoder = nn.ModuleList(
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in self.use_encoder_idx
        )

        num_blocks = round(3 * depth_mult)
        num_fusions = len(self.in_channels) - 1
        self.lateral_convs = nn.ModuleList(
            _conv_norm(hidden_dim, hidden_dim, 1, 1, act=act) for _ in range(num_fusions)
        )
        self.fpn_blocks = nn.ModuleList(
            CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks, expansion=expansion, act=act)
            for _ in range(num_fusions)
        )
        self.downsample_convs = nn.ModuleList(
            _conv_norm(hidden_dim, hidden_dim, 3, 2, act=act) for _ in range(num_fusions)
        )
        self.pan_blocks = nn.ModuleList(
            CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks, expansion=expansion, act=act)
            for _ in range(num_fusions)
        )

        # Cached embeddings are non-persistent: they stay out of checkpoints, so a model
        # built for another input size still loads them.
        self._pos_embed_sizes: dict[int, tuple[int, int]] = {}
        if eval_spatial_size is not None:
            eval_height, eval_width = eval_spatial_size
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                height, width = eval_height // stride, eval_width // stride
                self._pos_embed_sizes[idx] = (height, width)
                self.register_buffer(
                    f"pos_embed{idx}",
                    self.build_2d_sincos_position_embedding(
                        width, height, hidden_dim, pe_temperature
                    ),
                    persistent=False,
                )

    @staticmethod
    def build_2d_sincos_position_embedding(
        width: int, height: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> Float[Tensor, "1 num_tokens embed_dim"]:
        """Build the 2D sin-cos position embedding of a row-major flattened feature map.

        Args:
            width: Feature map width.
            height: Feature map height.
            embed_dim: Embedding dimension, divisible by 4.
            temperature: Frequency temperature.

        Returns:
            Embedding of the ``height * width`` tokens.
        """
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos embedding.")
        grid_h, grid_w = torch.meshgrid(
            torch.arange(int(height), dtype=torch.float32),
            torch.arange(int(width), dtype=torch.float32),
            indexing="ij",
        )
        pos_dim = embed_dim // 4
        omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None]

    def _position_embedding(
        self, idx: int, height: int, width: int, device: torch.device
    ) -> Float[Tensor, "1 num_tokens hidden_dim"]:
        """Return the position embedding of an encoder level, cached when the size matches.

        Args:
            idx: Input level index.
            height: Feature map height.
            width: Feature map width.
            device: Device of the feature map.

        Returns:
            Embedding of the ``height * width`` tokens.
        """
        if not self.training and self._pos_embed_sizes.get(idx) == (height, width):
            return getattr(self, f"pos_embed{idx}")
        return self.build_2d_sincos_position_embedding(
            width, height, self.hidden_dim, self.pe_temperature
        ).to(device)

    def forward(
        self, feats: Sequence[Float[Tensor, "batch_size channels height width"]]
    ) -> list[Float[Tensor, "batch_size hidden_dim height width"]]:
        """Fuse multi-scale backbone features.

        Args:
            feats: Backbone feature maps, one per entry of ``in_channels``.

        Returns:
            Fused feature maps with ``hidden_dim`` channels, at the input strides.
        """
        if len(feats) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} feature maps, got {len(feats)}.")
        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, feats)]

        # AIFI: self-attention within each level of use_encoder_idx (the coarsest by
        # default), written back in place of that level.
        for encoder, idx in zip(self.encoder, self.use_encoder_idx):
            height, width = proj_feats[idx].shape[2:]
            # (B, C, H, W) -> (B, H * W, C)
            src = proj_feats[idx].flatten(2).permute(0, 2, 1)
            pos_embed = self._position_embedding(idx, height, width, src.device)
            memory = encoder(src, pos_embed=pos_embed)
            proj_feats[idx] = (
                memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, height, width).contiguous()
            )

        # CCFF, top-down (FPN): inner_outs[0] is always the finest level fused so far.
        num_levels = len(self.in_channels)
        inner_outs = [proj_feats[-1]]
        for i, idx in enumerate(range(num_levels - 1, 0, -1)):
            feat_high = self.lateral_convs[i](inner_outs[0])
            inner_outs[0] = feat_high
            upsampled = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            inner_outs.insert(
                0, self.fpn_blocks[i](torch.concat([upsampled, proj_feats[idx - 1]], dim=1))
            )

        # CCFF, bottom-up (PAN).
        outs = [inner_outs[0]]
        for i in range(num_levels - 1):
            downsampled = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.concat([downsampled, inner_outs[i + 1]], dim=1)))
        return outs
