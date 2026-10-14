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

"""Unit tests for LearnedPositionalEncoding."""

import unittest

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.transfusions.positional_encoding import (
    LearnedPositionalEncoding,
)


class TestLearnedPositionalEncoding(unittest.TestCase):
    """Unit tests for the LearnedPositionalEncoding."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

        # BEV coordinates are (x, y), which is what the decoder layer feeds this module.
        self.input_channels = 2
        self.embed_dims = 16
        self.batch_size = 2
        self.num_positions = 6

        self.positional_encoding = self._build_positional_encoding()
        self.bev_positions = self._build_bev_positions()

    def _build_positional_encoding(self) -> LearnedPositionalEncoding:
        """Build the module under test on the shared device."""
        return LearnedPositionalEncoding(self.input_channels, self.embed_dims).to(self.device)

    def _build_bev_positions(
        self,
    ) -> Float32[torch.Tensor, "batch_size num_positions input_channels"]:
        """Build BEV cell coordinates spread over a plausible feature-map extent."""
        return (
            torch.rand(self.batch_size, self.num_positions, self.input_channels, device=self.device)
            * 16.0
        )

    def test_forward_maps_coordinates_to_the_embedding_width(self) -> None:
        """Test that the trailing coordinate axis is replaced by the embedding axis."""
        embeddings = self.positional_encoding(self.bev_positions)

        self.assertEqual(embeddings.shape, (self.batch_size, self.num_positions, self.embed_dims))
        self.assertTrue(torch.isfinite(embeddings).all())

    def test_forward_accepts_any_token_count(self) -> None:
        """
        Test that the module only cares about the trailing coordinate axis, so callers can hand it
        queries (batch, num_proposals, 2) or keys (batch, height*width, 2) alike.

        The Conv1d stack transposes the token and channel axes internally, so the input rank is
        fixed at three; only the token count is free.
        """
        query_positions = torch.rand(self.batch_size, 4, self.input_channels, device=self.device)
        key_positions = torch.rand(self.batch_size, 3 * 5, self.input_channels, device=self.device)

        self.assertEqual(
            self.positional_encoding(query_positions).shape,
            (self.batch_size, 4, self.embed_dims),
        )
        self.assertEqual(
            self.positional_encoding(key_positions).shape,
            (self.batch_size, 3 * 5, self.embed_dims),
        )

    def test_encodes_each_position_independently(self) -> None:
        """
        Test that a position's embedding depends only on its own coordinate. The decoder adds
        these to tokens before attention, so any mixing between positions here would leak one
        query's location into another's embedding.

        Moving a single position is what makes this bite: a term that mixes positions but stays
        permutation equivariant, such as adding the batch mean, would survive a permutation check.

        This holds at inference only. The BatchNorm1d normalizes over the batch and token axes, so
        while training every position's embedding does depend on the rest of the batch; eval mode
        swaps those batch statistics for the fixed running ones.
        """
        self.positional_encoding.eval()

        embeddings = self.positional_encoding(self.bev_positions)
        moved_positions = self.bev_positions.clone()
        moved_positions[0, 0] += 5.0

        moved_embeddings = self.positional_encoding(moved_positions)

        self.assertFalse(
            torch.allclose(moved_embeddings[0, 0], embeddings[0, 0], atol=1e-6),
            msg="moving a position left its own embedding unchanged",
        )
        self.assertTrue(
            torch.allclose(moved_embeddings[0, 1:], embeddings[0, 1:], atol=1e-6),
            msg="moving one position changed its neighbours' embeddings",
        )
        self.assertTrue(torch.allclose(moved_embeddings[1], embeddings[1], atol=1e-6))

    def test_distinct_positions_get_distinct_embeddings(self) -> None:
        """Test that the encoding actually varies with position rather than collapsing."""
        embeddings = self.positional_encoding(self.bev_positions)

        # Every pair of positions in the first sample differs somewhere in the embedding.
        for first in range(self.num_positions):
            for second in range(first + 1, self.num_positions):
                self.assertFalse(
                    torch.allclose(embeddings[0, first], embeddings[0, second], atol=1e-6),
                    msg=f"positions {first} and {second} collapsed to the same embedding",
                )

    def test_same_coordinate_gives_the_same_embedding(self) -> None:
        """Test that the encoding is a pure function of the coordinate, across batch elements."""
        repeated_positions = self.bev_positions[0:1].expand(
            3, self.num_positions, self.input_channels
        )

        embeddings = self.positional_encoding(repeated_positions.contiguous())

        self.assertTrue(torch.allclose(embeddings[0], embeddings[1], atol=1e-6))
        self.assertTrue(torch.allclose(embeddings[0], embeddings[2], atol=1e-6))

    def test_is_a_two_layer_projection_with_a_nonlinearity(self) -> None:
        """
        Test that the module is not a single affine map. A linear encoding could be folded into
        the attention projections and would not add anything.
        """
        modules = list(self.positional_encoding.proj)

        self.assertEqual(len(modules), 4)
        self.assertIsInstance(modules[0], nn.Conv1d)
        self.assertIsInstance(modules[1], nn.BatchNorm1d)
        self.assertIsInstance(modules[2], nn.ReLU)
        self.assertIsInstance(modules[3], nn.Conv1d)
        self.assertEqual(modules[0].in_channels, self.input_channels)
        self.assertEqual(modules[3].out_channels, self.embed_dims)
        # Kernel size one keeps every token's embedding a function of its own coordinate.
        self.assertEqual(modules[0].kernel_size, (1,))
        self.assertEqual(modules[3].kernel_size, (1,))

    def test_gradients_reach_the_positions_and_the_parameters(self) -> None:
        """Test that the encoding is differentiable, since it is learned alongside the decoder."""
        bev_positions = self.bev_positions.clone().requires_grad_(True)

        self.positional_encoding(bev_positions).sum().backward()

        assert bev_positions.grad is not None
        self.assertTrue(torch.isfinite(bev_positions.grad).all())
        for parameter in self.positional_encoding.parameters():
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
