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

"""Sparse 3D voxel middle encoder for BEVFusion.

SECOND/VoxelNet-style sparse 3D convolution backbone that consumes per-voxel features
and produces a dense BEV feature map. Built on the framework's export-aware spconv wrappers.

The sparse grid order is ``(Y, X, Z)`` (= the``SparseEncoder`` "(H, W, D)") so the final
``conv_out`` (stride on the last axis) collapses Z. The shared voxelization emits coordinates as
``[batch, z, y, x]``, so coordinates are reordered to ``[batch, y, x, z]`` here.
"""

from __future__ import annotations

from collections.abc import Sequence

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch import SparseConvTensor, SparseSequential
from spconv.pytorch.modules import SparseModule

# Native spconv conv layers are used for training (autograd-friendly). The
# framework's export wrappers in autoware_ml.ops.spconv are inference-only and
# would be swapped in at ONNX-export time.
SubMConv3d = spconv.SubMConv3d
SparseConv3d = spconv.SparseConv3d


def _norm(channels: int, eps: float, momentum: float) -> nn.BatchNorm1d:
    """Build the BatchNorm applied to sparse voxel features."""
    return nn.BatchNorm1d(channels, eps=eps, momentum=momentum)


class SparseBasicBlock(SparseModule):
    """Residual submanifold-convolution block (ResNet basic block, sparse).

    Inherits ``spconv``'s ``SparseModule`` so ``SparseSequential`` passes the
    full ``SparseConvTensor`` (not just its ``.features``).
    """

    def __init__(self, channels: int, indice_key: str, eps: float, momentum: float) -> None:
        super().__init__()
        self.conv1 = SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = _norm(channels, eps, momentum)
        self.conv2 = SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = _norm(channels, eps, momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Apply the residual sparse convolution block.

        Args:
            x: Input sparse convolution tensor.

        Returns:
            Output sparse tensor with the residual update applied.
        """
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity))
        return out


class SparseEncoder(nn.Module):
    """Sparse 3D voxel encoder producing a dense BEV feature map.

    Args:
        in_channels: Voxel feature channels (the voxel encoder's output dim).
        sparse_shape: Sparse grid shape in ``(Y, X, Z)`` order.
        base_channels: Channels after the input submanifold conv.
        encoder_channels: Per-stage channel tuples; the last entry of each
            non-final stage triggers a stride-2 downsample to that channel count.
        encoder_paddings: Per-stage paddings; the downsample conv of stage ``i``
            uses ``encoder_paddings[i][-1]`` (scalar or ``(Y, X, Z)`` tuple).
        output_channels: Channels of the final ``conv_out`` (before Z flatten).
        dense_output_shapes: Dense output shape ``(Y, X, Z)`` after ``conv_out``.
        norm_eps: BatchNorm epsilon.
        norm_momentum: BatchNorm momentum.
    """

    def __init__(
        self,
        in_channels: int,
        sparse_shape: Sequence[int],
        base_channels: int = 16,
        encoder_channels: Sequence[Sequence[int]] = (
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
            (128, 128),
        ),
        encoder_paddings: Sequence[Sequence] = ((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        output_channels: int = 128,
        dense_output_shapes: Sequence[int] = (180, 180, 2),
        norm_eps: float = 1e-3,
        norm_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.sparse_shape = list(sparse_shape)
        self.output_channels = output_channels
        self.dense_output_shapes = list(dense_output_shapes)
        num_stages = len(encoder_channels)

        self.conv_input = SparseSequential(
            SubMConv3d(
                in_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            ),
            _norm(base_channels, norm_eps, norm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encoder_layers = SparseSequential()
        current_channels = base_channels
        for stage_index, stage_channels in enumerate(encoder_channels):
            is_last_stage = stage_index == num_stages - 1
            blocks: list[nn.Module] = []
            for block_index, out_channels in enumerate(stage_channels):
                is_last_block = block_index == len(stage_channels) - 1
                if is_last_block and not is_last_stage:
                    # Stride-2 downsample to the next stage's channel count.
                    blocks.append(
                        SparseConv3d(
                            current_channels,
                            out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=encoder_paddings[stage_index][block_index],
                            bias=False,
                            indice_key=f"spconv{stage_index + 1}",
                        )
                    )
                    blocks.append(_norm(out_channels, norm_eps, norm_momentum))
                    blocks.append(nn.ReLU(inplace=True))
                    current_channels = out_channels
                else:
                    blocks.append(
                        SparseBasicBlock(
                            out_channels,
                            indice_key=f"subm{stage_index + 1}",
                            eps=norm_eps,
                            momentum=norm_momentum,
                        )
                    )
                    current_channels = out_channels
            self.encoder_layers.add_module(
                f"encoder_layer{stage_index + 1}", SparseSequential(*blocks)
            )

        self.conv_out = SparseSequential(
            SparseConv3d(
                current_channels,
                output_channels,
                kernel_size=(1, 1, 3),
                stride=(1, 1, 2),
                padding=0,
                bias=False,
                indice_key="spconv_down2",
            ),
            _norm(output_channels, norm_eps, norm_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode voxel features into a dense BEV map.

        Args:
            voxel_features: Per-voxel features of shape ``(N, in_channels)``.
            coords: Voxel coordinates of shape ``(N, 4)`` in ``[batch, z, y, x]``.
            batch_size: Number of samples in the batch.

        Returns:
            Dense BEV feature map of shape
            ``(batch_size, output_channels * Z, Y, X)``.
        """
        # Reorder [batch, z, y, x] -> [batch, y, x, z] to match sparse_shape (Y, X, Z).
        coords = coords[:, [0, 2, 3, 1]].contiguous().int()
        sp_tensor = SparseConvTensor(voxel_features, coords, self.sparse_shape, batch_size)
        x = self.conv_input(sp_tensor)
        x = self.encoder_layers(x)
        out = self.conv_out(x)
        return self._to_dense(out, batch_size)

    def _to_dense(self, sparse_tensor: SparseConvTensor, batch_size: int) -> torch.Tensor:
        """Scatter the sparse output into a dense ``(B, C*Z, Y, X)`` tensor."""
        height, width, depth = self.dense_output_shapes  # (Y, X, Z)
        features = sparse_tensor.features
        channels = features.shape[1]
        indices = sparse_tensor.indices.to(features.device).long()
        b, h, w, d = indices.unbind(1)
        linear_idx = ((b * height + h) * width + w) * depth + d
        dense = features.new_zeros((batch_size * height * width * depth, channels))
        dense.scatter_(0, linear_idx.unsqueeze(1).expand(-1, channels), features)
        dense = dense.view(batch_size, height, width, depth, channels)
        # (B, Y, X, Z, C) -> (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        dense = dense.permute(0, 4, 3, 1, 2).contiguous()
        return dense.view(batch_size, channels * depth, height, width)
