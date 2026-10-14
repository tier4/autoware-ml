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

"""Unit tests for TransFusionDecoderLayer."""

import unittest

from jaxtyping import Float32
import torch

from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_decoder_layer import (
    TransFusionDecoderLayer,
)


class TestTransFusionDecoderLayer(unittest.TestCase):
    """Unit tests for the TransFusionDecoderLayer."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

        self.embed_dims = 16
        self.num_heads = 4
        self.feedforward_channels = 32
        self.dropout = 0.1
        self.batch_size = 2
        self.num_proposals = 5
        # Keys are the flattened BEV map, so they are longer than the queries.
        self.num_keys = 9

        self.decoder_layer = self._build_decoder_layer()
        self.query = self._build_features(self.num_proposals)
        self.key = self._build_features(self.num_keys)
        self.query_pos = self._build_positions(self.num_proposals)
        self.key_pos = self._build_positions(self.num_keys)

    def _build_decoder_layer(self) -> TransFusionDecoderLayer:
        """Build the layer under test in eval mode, so dropout cannot blur the comparisons."""
        return (
            TransFusionDecoderLayer(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                feedforward_channels=self.feedforward_channels,
                dropout=self.dropout,
            )
            .to(self.device)
            .eval()
        )

    def _build_features(
        self, sequence_length: int
    ) -> Float32[torch.Tensor, "batch_size embed_dims sequence_length"]:
        """Build a channels-first feature tensor, which is the layout the layer takes."""
        return torch.randn(self.batch_size, self.embed_dims, sequence_length, device=self.device)

    def _build_positions(
        self, sequence_length: int
    ) -> Float32[torch.Tensor, "batch_size sequence_length 2"]:
        """Build BEV (x, y) coordinates spread over a plausible feature-map extent."""
        return torch.rand(self.batch_size, sequence_length, 2, device=self.device) * 16.0

    def _forward(
        self,
        query: Float32[torch.Tensor, "batch_size embed_dims num_proposals"] | None = None,
        query_pos: Float32[torch.Tensor, "batch_size num_proposals 2"] | None = None,
        key_pos: Float32[torch.Tensor, "batch_size num_keys 2"] | None = None,
    ) -> Float32[torch.Tensor, "batch_size embed_dims num_proposals"]:
        """Run the layer, defaulting every argument to the setUp fixture."""
        return self.decoder_layer(
            query=self.query if query is None else query,
            key=self.key,
            query_pos=self.query_pos if query_pos is None else query_pos,
            key_pos=self.key_pos if key_pos is None else key_pos,
        )

    def test_forward_keeps_the_channels_first_query_layout(self) -> None:
        """
        Test that the layer returns the query in the layout it received. It transposes to tokens
        internally, so a missing transpose back would silently reshape the whole decoder stack.
        """
        refined_query = self._forward()

        self.assertEqual(refined_query.shape, self.query.shape)
        self.assertTrue(torch.isfinite(refined_query).all())

    def test_forward_is_deterministic_in_eval_mode(self) -> None:
        """Test that dropout is inactive in eval mode, so two calls agree exactly."""
        self.assertTrue(torch.equal(self._forward(), self._forward()))

    def test_forward_applies_dropout_only_while_training(self) -> None:
        """Test that the dropout the layer is configured with is active in train mode."""
        self.decoder_layer.train()

        torch.manual_seed(1)
        first = self._forward()
        torch.manual_seed(2)
        second = self._forward()

        self.assertFalse(torch.equal(first, second))

    def test_forward_is_equivariant_to_the_query_order(self) -> None:
        """
        Test that permuting the queries permutes the outputs the same way.

        The head selects proposals with an unsorted topk and slices the decoder's output by layer,
        so nothing downstream fixes a query to a position. That only holds because this layer
        treats the query axis symmetrically, with no index-based positional term.
        """
        refined_query = self._forward()
        permutation = torch.randperm(self.num_proposals, device=self.device)

        permuted_query = self._forward(
            query=self.query[:, :, permutation], query_pos=self.query_pos[:, permutation]
        )

        self.assertTrue(torch.allclose(permuted_query, refined_query[:, :, permutation], atol=1e-5))

    def test_forward_depends_on_the_query_positions(self) -> None:
        """Test that query positions reach the output, so the encoding is not dead weight."""
        refined_query = self._forward()

        moved_query = self._forward(query_pos=self.query_pos + 5.0)

        self.assertFalse(torch.allclose(moved_query, refined_query, atol=1e-5))

    def test_forward_depends_on_the_key_positions(self) -> None:
        """Test that key positions reach the output through the cross attention."""
        refined_query = self._forward()

        moved_key = self._forward(key_pos=self.key_pos + 5.0)

        self.assertFalse(torch.allclose(moved_key, refined_query, atol=1e-5))

    def test_gradients_reach_the_inputs_and_the_parameters(self) -> None:
        """Test that the layer stays differentiable end to end."""
        query = self.query.clone().requires_grad_(True)

        self._forward(query=query).sum().backward()

        assert query.grad is not None
        self.assertTrue(torch.isfinite(query.grad).all())
        for parameter in self.decoder_layer.parameters():
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_forward_is_unchanged_by_the_exportable_attention_swap(self) -> None:
        """
        Test that swapping both attentions for their exportable equivalents leaves the layer's
        output alone, which is what the export path relies on when it rewrites these modules.
        """
        refined_query = self._forward()

        self.decoder_layer.self_attn = ExportableMultiheadAttention(  # type: ignore
            self.decoder_layer.self_attn
        ).eval()
        self.decoder_layer.cross_attn = ExportableMultiheadAttention(  # type: ignore
            self.decoder_layer.cross_attn
        ).eval()
        exported_query = self._forward()

        self.assertTrue(torch.allclose(exported_query, refined_query, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
