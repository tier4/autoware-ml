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

import unittest

from jaxtyping import Float
import torch
from torch import Tensor

from autoware_ml.models.common.layers.ms_deformable_attention import (
    MSDeformableAttention,
    deformable_attention_core,
)

# A non-square level, so a swapped x and y would land elsewhere.
HEIGHT, WIDTH = 4, 8


def _ramp() -> Float[Tensor, "1 num_values 1 2"]:
    """Values of a 4 x 8 level whose two channels are the column and the row of each pixel.

    Bilinear sampling inside the map then returns the sampled pixel coordinates exactly.
    """
    rows, cols = torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float32),
        torch.arange(WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([cols, rows], dim=-1).reshape(1, HEIGHT * WIDTH, 1, 2)


class TestDeformableAttentionCore(unittest.TestCase):
    """The sampling and weighting of the deformable attention."""

    def test_samples_x_along_the_width_and_y_along_the_height(self) -> None:
        """
        Input: the coordinate ramp of a 4 x 8 level, sampled at normalized (x, y) =
            (0.6, 0.45) with weight 1.
        Expected: column 0.6 * 8 - 0.5 = 4.3 and row 0.45 * 4 - 0.5 = 1.3, the pixel centers
            sitting at half-pixel offsets.
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
        Expected: zero, as with grid_sample's zero padding.
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
        Input: a 4 x 8 level of ones followed by a 2 x 4 level of threes, one point on each
            with weights 0.25 and 0.75.
        Expected: 0.25 * 1 + 0.75 * 3 = 2.5, so the values and the points are split by level
            in the same order.
        Check: the output.
        """
        value = torch.concat([torch.ones(1, 32, 1, 2), torch.full((1, 8, 1, 2), 3.0)], dim=1)

        out = deformable_attention_core(
            value,
            [(4, 8), (2, 4)],
            torch.full((1, 1, 1, 2, 2), 0.5),
            torch.tensor([0.25, 0.75]).reshape(1, 1, 1, 2),
            [1, 1],
        )

        torch.testing.assert_close(out, torch.full((1, 1, 2), 2.5))


class TestMSDeformableAttention(unittest.TestCase):
    """The deformable attention module."""

    def test_offsets_scale_with_the_box_and_the_point_count(self) -> None:
        """
        Input: a single-head attention over the coordinate ramp, with identity projections,
            the learned offsets of its two points set to (1, -1) and (-2, 2), and the weights
            selecting one point at a time; a reference box (0.5, 0.5, 0.4, 0.2).
        Expected: point k sits at the box center plus offset_k / 2 points * box size * 0.5,
            (0.6, 0.45) and (0.3, 0.6), so it samples column 4.3 and row 1.3, then column 1.9
            and row 1.9.
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
                        torch.zeros(1, 1, 2), reference, _ramp().reshape(1, -1, 2), [(4, 8)]
                    )

                torch.testing.assert_close(out, torch.tensor([[expected]]))

    def test_initial_points_spread_outwards_with_uniform_weights(self) -> None:
        """
        Input: a fresh attention of 4 heads with 2 points on one level.
        Expected: head h points along angle 2 pi h / 4 scaled so its larger coordinate is 1,
            the second point twice as far as the first, and the weights start uniform.
        Check: the offset biases, and the zero weights of the offset and weight layers.
        """
        attention = MSDeformableAttention(embed_dim=8, num_heads=4, num_levels=1, num_points=2)

        offsets = attention.sampling_offsets.bias.reshape(4, 2, 2)
        directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        torch.testing.assert_close(offsets[:, 0], directions, atol=1e-6, rtol=0)
        torch.testing.assert_close(offsets[:, 1], 2 * directions, atol=1e-6, rtol=0)
        self.assertEqual(attention.sampling_offsets.weight.abs().sum().item(), 0.0)
        self.assertEqual(attention.attention_weights.weight.abs().sum().item(), 0.0)
        self.assertEqual(attention.attention_weights.bias.abs().sum().item(), 0.0)

    def test_rejects_point_counts_and_dimensions_that_do_not_fit(self) -> None:
        """
        Input: two point counts for three levels, and 30 channels for 8 heads.
        Expected: each is refused at construction.
        Check: the ValueError.
        """
        for kwargs in (
            {"num_levels": 3, "num_points": [4, 4]},
            {"embed_dim": 30, "num_heads": 8},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MSDeformableAttention(**kwargs)

    def test_state_dict_follows_rt_detr_v2_checkpoints(self) -> None:
        """
        Input: the attention of the RT-DETRv2 decoder layers: 256 channels, 8 heads, 3
            levels of 4 points.
        Expected: the entries and shapes of the cross-attention of RT-DETRv2 checkpoints
            trained with the official code, the point scale persistent as there.
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

    def test_gradients_reach_every_parameter_the_queries_and_the_values(self) -> None:
        """
        Input: an attention with 3 points on the first level and 1 on the second, its offset
            and weight layers randomized as after training, random queries, boxes and values
            that require gradients.
        Expected: a loss on the output reaches every parameter, the queries and the values.
            (Freshly initialized, the offsets and weights ignore the queries.)
        Check: every gradient is present and non-zero.
        """
        torch.manual_seed(0)
        attention = MSDeformableAttention(
            embed_dim=16, num_heads=2, num_levels=2, num_points=[3, 1]
        )
        torch.nn.init.normal_(attention.sampling_offsets.weight, std=0.1)
        torch.nn.init.normal_(attention.attention_weights.weight, std=0.1)
        query = torch.randn(2, 5, 16, requires_grad=True)
        value = torch.randn(2, 4 * 8 + 2 * 4, 16, requires_grad=True)
        reference = torch.rand(2, 5, 1, 4) * 0.5 + 0.25

        out = attention(query, reference, value, [(4, 8), (2, 4)])
        (out * torch.randn_like(out)).sum().backward()

        for name, param in attention.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(param.grad.abs().sum().item(), 0.0, name)
        self.assertGreater(query.grad.abs().sum().item(), 0.0)
        self.assertGreater(value.grad.abs().sum().item(), 0.0)
