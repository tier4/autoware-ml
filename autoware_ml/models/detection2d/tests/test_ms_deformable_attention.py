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

"""Unit tests for the multi-scale deformable attention of RT-DETRv2."""

from __future__ import annotations

import itertools
import math
import unittest

from jaxtyping import Bool, Float
import torch
from torch import Tensor

from autoware_ml.models.detection2d.layers.ms_deformable_attention import (
    MSDeformableAttention,
    deformable_attention_core,
)

# A non-square level, so a swapped x and y would land elsewhere.
HEIGHT, WIDTH = 4, 8


def _ramp(height: int = HEIGHT, width: int = WIDTH) -> Float[Tensor, "1 num_values 1 2"]:
    """Values of a level whose two channels are the column and the row of each pixel.

    Bilinear sampling inside the map then returns the sampled pixel coordinates exactly.
    """
    rows, cols = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([cols, rows], dim=-1).reshape(1, height * width, 1, 2)


def _bilinear(
    level: Float[Tensor, "channels height width"], x: float, y: float
) -> Float[Tensor, " channels"]:
    """Sample a level at normalized ``(x, y)`` pixel by pixel, zero outside the map."""
    channels, height, width = level.shape
    col, row = x * width - 0.5, y * height - 0.5
    col0, row0 = math.floor(col), math.floor(row)
    sample = torch.zeros(channels)
    for r, c in ((row0, col0), (row0, col0 + 1), (row0 + 1, col0), (row0 + 1, col0 + 1)):
        if 0 <= r < height and 0 <= c < width:
            sample += level[:, r, c] * (1 - abs(row - r)) * (1 - abs(col - c))
    return sample


def _reference_attention(
    attention: MSDeformableAttention,
    query: Float[Tensor, "batch_size num_queries embed_dim"],
    reference_points: Float[Tensor, "batch_size num_queries 1 4"],
    value: Float[Tensor, "batch_size num_values embed_dim"],
    shapes: list[tuple[int, int]],
    value_mask: Bool[Tensor, "batch_size num_values"],
) -> Float[Tensor, "batch_size num_queries embed_dim"]:
    """The attention computed one batch entry, query, head and point at a time."""
    batch_size, num_queries, _ = query.shape
    num_heads, head_dim = attention.num_heads, attention.head_dim
    points = attention.num_points_list
    projected = attention.value_proj(value) * value_mask[..., None]
    offsets = attention.sampling_offsets(query).reshape(
        batch_size, num_queries, num_heads, sum(points), 2
    )
    weights = attention.attention_weights(query).reshape(
        batch_size, num_queries, num_heads, sum(points)
    )
    weights = weights.softmax(-1)
    starts = [0, *itertools.accumulate(height * width for height, width in shapes)]
    out = torch.zeros(batch_size, num_queries, num_heads * head_dim)
    for b, q, h in itertools.product(range(batch_size), range(num_queries), range(num_heads)):
        center, size = reference_points[b, q, 0, :2], reference_points[b, q, 0, 2:]
        point = 0
        for level, ((height, width), count) in enumerate(zip(shapes, points, strict=True)):
            channels = slice(h * head_dim, (h + 1) * head_dim)
            level_value = projected[b, starts[level] : starts[level + 1], channels]
            level_value = level_value.T.reshape(head_dim, height, width)
            for _ in range(count):
                x, y = center + offsets[b, q, h, point] / count * size * attention.offset_scale
                out[b, q, channels] += weights[b, q, h, point] * _bilinear(level_value, x, y)
                point += 1
    return attention.output_proj(out)


class TestDeformableAttentionCore(unittest.TestCase):
    """The sampling and weighting of the deformable attention."""

    def test_samples_x_along_the_width_and_y_along_the_height(self) -> None:
        """
        Input: the coordinate ramp of a 4 x 8 level, sampled at normalized (x, y) =
        (0.6, 0.45) with weight 1.
        Expected: column 0.6 * 8 - 0.5 = 4.3 and row 0.45 * 4 - 0.5 = 1.3, since pixel centers
        sit at half-pixel offsets.
        Check: the two output channels.
        """
        out = deformable_attention_core(
            _ramp(),
            [(HEIGHT, WIDTH)],
            torch.tensor([0.6, 0.45]).reshape(1, 1, 1, 1, 2),
            torch.ones(1, 1, 1, 1),
            [1],
        )

        torch.testing.assert_close(out, torch.tensor([[[4.3, 1.3]]]))

    def test_points_outside_the_map_sample_zeros(self) -> None:
        """
        Input: a level of ones, sampled at a point beyond its right edge with weight 1.
        Expected: zero, as with the zero padding of ``grid_sample``.
        Check: the output.
        """
        out = deformable_attention_core(
            torch.ones(1, HEIGHT * WIDTH, 1, 2),
            [(HEIGHT, WIDTH)],
            torch.tensor([1.5, 0.5]).reshape(1, 1, 1, 1, 2),
            torch.ones(1, 1, 1, 1),
            [1],
        )

        torch.testing.assert_close(out, torch.zeros(1, 1, 2))

    def test_each_level_takes_its_own_points_and_weights(self) -> None:
        """
        Input: the coordinate ramps of a 4 x 8 and a 2 x 4 level, 2 points on the first at
        (0.6, 0.45) and (0.3, 0.6), 1 on the second at (0.5, 0.75), with weights 0.2, 0.3 and
        0.5.
        Expected: the points sample (column, row) = (4.3, 1.3), (1.9, 1.9) and (1.5, 1.0), so
        the weighted sum is (2.18, 1.33). It holds only if the values, the points and the
        weights are all split by level.
        Check: the output.
        """
        value = torch.concat([_ramp(), _ramp(2, 4)], dim=1)
        locations = torch.tensor([[0.6, 0.45], [0.3, 0.6], [0.5, 0.75]]).reshape(1, 1, 1, 3, 2)

        out = deformable_attention_core(
            value,
            [(HEIGHT, WIDTH), (2, 4)],
            locations,
            torch.tensor([0.2, 0.3, 0.5]).reshape(1, 1, 1, 3),
            [2, 1],
        )

        torch.testing.assert_close(out, torch.tensor([[[2.18, 1.33]]]))


class TestMSDeformableAttention(unittest.TestCase):
    """The deformable attention module."""

    def test_offsets_scale_with_the_box_and_the_point_count(self) -> None:
        """
        Input: a single-head attention over the coordinate ramp with identity projections and
        two points, whose offsets are set to (1, -1) and (-2, 2). The weights select one
        point at a time. The reference box is (0.5, 0.5, 0.4, 0.2).
        Expected: point k sits at center + offset_k * box size * 0.5 / ``num_points``, that is
        (0.6, 0.45) and (0.3, 0.6). They sample (column, row) = (4.3, 1.3) and (1.9, 1.9).
        Check: the output for each selected point.
        """
        attention = MSDeformableAttention(embed_dim=2, num_heads=1, num_levels=1, num_points=2)
        with torch.no_grad():
            for proj in (attention.value_proj, attention.output_proj):
                proj.weight.copy_(torch.eye(2))
                proj.bias.zero_()
            attention.sampling_offsets.bias.copy_(torch.tensor([1.0, -1.0, -2.0, 2.0]))
        reference = torch.tensor([0.5, 0.5, 0.4, 0.2]).reshape(1, 1, 1, 4)

        for point, expected in ((0, [4.3, 1.3]), (1, [1.9, 1.9])):
            with self.subTest(point=point):
                with torch.no_grad():
                    selection = torch.full((2,), -1e4)
                    selection[point] = 0.0
                    attention.attention_weights.bias.copy_(selection)

                    out = attention(
                        torch.zeros(1, 1, 2),
                        reference,
                        _ramp().reshape(1, -1, 2),
                        [(HEIGHT, WIDTH)],
                    )

                torch.testing.assert_close(out, torch.tensor([[expected]]))

    def test_initial_points_spread_outwards_with_uniform_weights(self) -> None:
        """
        Input: a fresh attention of 8 heads with 2 points on the first level and 1 on the
        second.
        Expected: head h points along angle 2 pi h / 8, scaled so that its larger coordinate
        is 1, so the diagonal heads point at (1, 1) and not (0.71, 0.71); on each level, the
        k-th point is k times as far as the first; the weights start uniform.
        Check: the offset biases, and the zero weights of the offset and weight layers.
        """
        attention = MSDeformableAttention(
            embed_dim=16, num_heads=8, num_levels=2, num_points=[2, 1]
        )

        offsets = attention.sampling_offsets.bias.reshape(8, 3, 2)
        directions = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]]
            + [[-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]]
        )
        distances = torch.tensor([1.0, 2.0, 1.0]).reshape(1, 3, 1)
        torch.testing.assert_close(offsets, directions[:, None] * distances, atol=1e-5, rtol=0)
        self.assertEqual(attention.sampling_offsets.weight.abs().sum().item(), 0.0)
        self.assertEqual(attention.attention_weights.weight.abs().sum().item(), 0.0)
        self.assertEqual(attention.attention_weights.bias.abs().sum().item(), 0.0)

    def test_rejects_wrong_point_counts_and_channels_not_divisible_by_heads(self) -> None:
        """
        Input: two point counts for three levels; and 30 channels for 8 heads.
        Expected: each is refused at construction.
        Check: a ValueError is raised for each.
        """
        for kwargs in (
            {"num_levels": 3, "num_points": [4, 4]},
            {"embed_dim": 30, "num_heads": 8},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MSDeformableAttention(**kwargs)

    def test_rejects_other_level_counts_and_non_box_reference_points(self) -> None:
        """
        Input: an attention of 2 levels given the shapes of 3 levels; and reference points
        given as (x, y) points, as one box per level, and without the level axis.
        Expected: each is refused rather than sampled; without the check, a third level would
        be dropped silently.
        Check: a ValueError is raised for each.
        """
        attention = MSDeformableAttention(embed_dim=8, num_heads=2, num_levels=2, num_points=1)
        shapes = [(HEIGHT, WIDTH), (2, 4)]
        query = torch.zeros(1, 3, 8)
        value = torch.zeros(1, HEIGHT * WIDTH + 2 * 4 + 1, 8)
        box = torch.full((1, 3, 1, 4), 0.5)
        for reference, level_shapes in (
            (box, [*shapes, (1, 1)]),
            (box[..., :2], shapes),
            (box.expand(1, 3, 2, 4), shapes),
            (box[:, :, 0], shapes),
        ):
            with (
                self.subTest(reference=tuple(reference.shape), levels=len(level_shapes)),
                self.assertRaises(ValueError),
            ):
                attention(query, reference, value, level_shapes)

    def test_matches_a_point_by_point_reference(self) -> None:
        """
        Input: 2 images, 3 queries, 2 heads, a 4 x 8 level of 3 points and a 2 x 4 level of 1,
        ``offset_scale=0.7``, random weights including the biases of the projections, and a
        random ``value_mask``.
        Expected: the output of a reference that samples one batch entry, query, head and
        point at a time with explicit bilinear weights, so the layout of the batch, heads and
        points, the scaling of the offsets and the order of the mask and the value projection
        all follow the definition.
        Check: the output against the reference.
        """
        torch.manual_seed(0)
        attention = MSDeformableAttention(
            embed_dim=8, num_heads=2, num_levels=2, num_points=[3, 1], offset_scale=0.7
        )
        with torch.no_grad():
            for layer in (attention.sampling_offsets, attention.attention_weights):
                layer.weight.normal_(std=0.1)
            attention.value_proj.bias.normal_()
            attention.output_proj.bias.normal_()
        shapes = [(HEIGHT, WIDTH), (2, 4)]
        query = torch.randn(2, 3, 8)
        reference = torch.rand(2, 3, 1, 4) * torch.tensor([0.6, 0.6, 0.4, 0.4]) + 0.2
        value = torch.randn(2, HEIGHT * WIDTH + 2 * 4, 8)
        value_mask = torch.rand(2, HEIGHT * WIDTH + 2 * 4) > 0.3

        with torch.no_grad():
            out = attention(query, reference, value, shapes, value_mask)
            expected = _reference_attention(attention, query, reference, value, shapes, value_mask)

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_state_dict_follows_rt_detr_v2_checkpoints(self) -> None:
        """
        Input: the attention of the RT-DETRv2 decoder layers: 256 channels, 8 heads, 3
        levels of 4 points.
        Expected: the entries and shapes of the cross-attention in RT-DETRv2 checkpoints of
        the official code, including the persistent ``num_points_scale`` buffer.
        Check: the state dict shapes.
        """
        attention = MSDeformableAttention(embed_dim=256, num_heads=8, num_levels=3, num_points=4)

        self.assertEqual(
            {key: tuple(value.shape) for key, value in attention.state_dict().items()},
            {
                "num_points_scale": (12,),
                "sampling_offsets.weight": (192, 256),
                "sampling_offsets.bias": (192,),
                "attention_weights.weight": (96, 256),
                "attention_weights.bias": (96,),
                "value_proj.weight": (256, 256),
                "value_proj.bias": (256,),
                "output_proj.weight": (256, 256),
                "output_proj.bias": (256,),
            },
        )

    def test_value_mask_zeroes_the_padding(self) -> None:
        """
        Input: a fresh attention over two levels and random values, the right half of every
        level marked as padding by ``value_mask``.
        Expected: the output equals that of values whose padding is zero, since the fresh value
        projection has a zero bias, and differs from the output without the mask.
        Check: the masked output against both.
        """
        torch.manual_seed(0)
        attention = MSDeformableAttention(embed_dim=16, num_heads=2, num_levels=2, num_points=2)
        shapes = [(HEIGHT, WIDTH), (2, 4)]
        query = torch.randn(1, 5, 16)
        reference = torch.tensor([0.5, 0.5, 0.8, 0.8]).expand(1, 5, 1, 4)
        value = torch.randn(1, HEIGHT * WIDTH + 2 * 4, 16)
        value_mask = torch.concat(
            [(torch.arange(width) < width // 2).tile(height) for height, width in shapes]
        )[None]

        masked = attention(query, reference, value, shapes, value_mask)

        zeroed = value * value_mask[..., None]
        torch.testing.assert_close(masked, attention(query, reference, zeroed, shapes))
        self.assertFalse(torch.allclose(masked, attention(query, reference, value, shapes)))

    def test_gradients_reach_every_parameter_the_queries_and_the_values(self) -> None:
        """
        Input: an attention with 3 points on the first level and 1 on the second, and random
        queries, boxes and values that require gradients. The offset and weight layers are
        randomized: freshly initialized, their weights are zero, so they ignore the queries.
        Expected: a loss on the output reaches every parameter, the queries and the values.
        Check: every gradient is present and non-zero.
        """
        torch.manual_seed(0)
        attention = MSDeformableAttention(
            embed_dim=16, num_heads=2, num_levels=2, num_points=[3, 1]
        )
        torch.nn.init.normal_(attention.sampling_offsets.weight, std=0.1)
        torch.nn.init.normal_(attention.attention_weights.weight, std=0.1)
        query = torch.randn(2, 5, 16, requires_grad=True)
        value = torch.randn(2, HEIGHT * WIDTH + 2 * 4, 16, requires_grad=True)
        reference = torch.rand(2, 5, 1, 4) * 0.5 + 0.25

        out = attention(query, reference, value, [(HEIGHT, WIDTH), (2, 4)])
        (out * torch.randn_like(out)).sum().backward()

        for name, param in attention.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(param.grad.abs().sum().item(), 0.0, name)
        self.assertGreater(query.grad.abs().sum().item(), 0.0)
        self.assertGreater(value.grad.abs().sum().item(), 0.0)
