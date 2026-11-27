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

"""Multi-scale deformable attention of RT-DETRv2, ported from the official code
(lyuwenyu/RT-DETR@df2a96c): rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py and
rtdetrv2_pytorch/src/zoo/rtdetr/utils.py.

Every query attends to a few points per head and level, placed around its reference box by
learned offsets, instead of to every token. Sampling uses ``F.grid_sample`` (the "default"
method of the official code), which exports to ONNX and TensorRT without a custom op. The
"discrete" method is not ported.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

from jaxtyping import Bool, Float
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def deformable_attention_core(
    value: Float[Tensor, "batch_size num_values num_heads head_dim"],
    value_spatial_shapes: Sequence[Sequence[int]],
    sampling_locations: Float[Tensor, "batch_size num_queries num_heads total_points 2"],
    attention_weights: Float[Tensor, "batch_size num_queries num_heads total_points"],
    num_points_list: Sequence[int],
) -> Float[Tensor, "batch_size num_queries embed_dim"]:
    """Sample every level with bilinear interpolation and sum the samples by their weights.

    Args:
        value: Values of every level, flattened row by row and concatenated level by level.
        value_spatial_shapes: ``(height, width)`` of each level, in the order of ``value``.
        sampling_locations: ``(x, y)`` of every sampling point, normalized to [0, 1] by the
            level size; the points of each level follow those of the previous one. Points
            outside the map sample zeros.
        attention_weights: Weight of every sampling point.
        num_points_list: Number of sampling points on each level.

    Returns:
        Weighted sums of the samples, the heads concatenated.
    """
    batch_size, _, num_heads, head_dim = value.shape
    num_queries = sampling_locations.shape[1]

    split_shape = [height * width for height, width in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    # Map the [0, 1] locations to the [-1, 1] grid of grid_sample.
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grid_list = sampling_grids.split(list(num_points_list), dim=-2)

    sampled = [
        F.grid_sample(
            value_level.reshape(batch_size * num_heads, head_dim, height, width),
            grid_level,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        for value_level, grid_level, (height, width) in zip(
            value_list, sampling_grid_list, value_spatial_shapes, strict=True
        )
    ]
    weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (torch.concat(sampled, dim=-1) * weights).sum(-1)
    return output.reshape(batch_size, num_heads * head_dim, num_queries).permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention around box-shaped reference points.

    Each head of each query samples ``num_points`` points per level at the center of its
    reference box plus a learned offset. The offsets scale with the box size, so a query
    looks inside and around its box. Learned weights, normalized by a softmax over all the
    points of a head, sum the samples.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int | Sequence[int] = 4,
        offset_scale: float = 0.5,
    ) -> None:
        """Initialize the attention.

        Args:
            embed_dim: Query and value dimension.
            num_heads: Number of attention heads.
            num_levels: Number of feature levels.
            num_points: Sampling points per head and level, or one count per level.
            offset_scale: Scale of the offsets: a learned offset ``o`` moves a point by
                ``o * offset_scale / num_points`` box sizes, where ``num_points`` is the point
                count of its level.

        Raises:
            ValueError: If ``num_points`` does not give one count per level, or ``embed_dim``
                is not divisible by ``num_heads``.
        """
        super().__init__()
        if isinstance(num_points, int):
            num_points_list = [num_points] * num_levels
        else:
            num_points_list = [int(n) for n in num_points]
            if len(num_points_list) != num_levels:
                raise ValueError(
                    f"num_points must give one count per level ({num_levels}), "
                    f"got {num_points_list}."
                )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        self.num_points_list = num_points_list
        self.head_dim = embed_dim // num_heads
        total_points = sum(num_points_list)

        # Persistent, as in the official code, so its checkpoints load with strict=True.
        self.register_buffer(
            "num_points_scale",
            torch.tensor([1 / n for n in num_points_list for _ in range(n)], dtype=torch.float32),
        )

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Start with uniform weights and points spread outwards, one direction per head."""
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        # The k-th point of a level starts k times as far out as the first.
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list])
        grid_init *= scaling.reshape(1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.flatten())

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: Float[Tensor, "batch_size num_queries embed_dim"],
        reference_points: Float[Tensor, "batch_size num_queries 1 4"],
        value: Float[Tensor, "batch_size num_values embed_dim"],
        value_spatial_shapes: Sequence[Sequence[int]],
        value_mask: Bool[Tensor, "batch_size num_values"] | None = None,
    ) -> Float[Tensor, "batch_size num_queries embed_dim"]:
        """Attend to the multi-scale values around the reference box of each query.

        Args:
            query: Queries.
            reference_points: ``(cx, cy, w, h)`` reference box of every query, normalized by
                the image size.
            value: Values of every level, flattened row by row and concatenated level by
                level.
            value_spatial_shapes: ``(height, width)`` of each level, in the order of ``value``.
            value_mask: True for the values of the image, False for those of its padding,
                which are zeroed. None when the inputs are not padded.

        Returns:
            Attended features.

        Raises:
            ValueError: If ``value_spatial_shapes`` does not give one shape per level, or the
                reference points are not one box per query.
        """
        if len(value_spatial_shapes) != self.num_levels:
            raise ValueError(
                f"value_spatial_shapes must give one shape per level ({self.num_levels}), "
                f"got {len(value_spatial_shapes)}."
            )
        if reference_points.shape[-2:] != (1, 4):
            raise ValueError(
                "reference_points must hold one (cx, cy, w, h) box per query, got shape "
                f"{tuple(reference_points.shape)}."
            )
        batch_size, num_queries = query.shape[:2]
        total_points = sum(self.num_points_list)

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).reshape(
            batch_size, num_queries, self.num_heads, total_points, 2
        )
        attention_weights = F.softmax(
            self.attention_weights(query).reshape(
                batch_size, num_queries, self.num_heads, total_points
            ),
            dim=-1,
        )

        num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        offset = (
            sampling_offsets
            * num_points_scale
            * reference_points[:, :, None, :, 2:]
            * self.offset_scale
        )
        sampling_locations = reference_points[:, :, None, :, :2] + offset

        output = deformable_attention_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )
        return self.output_proj(output)
