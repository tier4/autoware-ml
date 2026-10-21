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

"""Unit tests for ExportableMultiheadAttention."""

import unittest

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)


class TestExportableMultiheadAttention(unittest.TestCase):
    """Unit tests for the ExportableMultiheadAttention."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

        self.embed_dims = 16
        self.num_heads = 4
        self.dropout = 0.1
        self.batch_size = 2
        self.num_queries = 5
        # A different key length than query length, so a shape mistake cannot pass unnoticed and
        # the cross-attention case is covered by default.
        self.num_keys = 7

        self.attention = self._build_attention()
        self.exportable_attention = ExportableMultiheadAttention(self.attention).eval()
        self.query = self._build_tokens(self.num_queries)
        self.key = self._build_tokens(self.num_keys)
        self.value = self._build_tokens(self.num_keys)

    def _build_attention(self, batch_first: bool = True) -> nn.MultiheadAttention:
        """Build a MultiheadAttention with non-trivial weights to copy from."""
        attention = nn.MultiheadAttention(
            self.embed_dims, self.num_heads, dropout=self.dropout, batch_first=batch_first
        ).to(self.device)
        # The default init leaves the biases at zero, which would hide a bias copy that is
        # dropped or swapped between the q, k and v chunks.
        with torch.no_grad():
            attention.in_proj_bias.normal_()
            attention.out_proj.bias.normal_()
        return attention.eval()

    def _build_tokens(
        self, sequence_length: int
    ) -> Float32[torch.Tensor, "batch_size sequence_length embed_dims"]:
        """Build a batch-first token tensor of the module's embedding width."""
        return torch.randn(self.batch_size, sequence_length, self.embed_dims, device=self.device)

    def test_matches_torch_multihead_attention_for_self_attention(self) -> None:
        """
        Test that the exported module reproduces nn.MultiheadAttention on self attention, which is
        the whole contract: the graph changes but the math must not.
        """
        expected, _ = self.attention(self.query, self.query, self.query, need_weights=False)

        attended, _ = self.exportable_attention(self.query, self.query, self.query)

        self.assertTrue(torch.allclose(attended, expected, atol=1e-5))

    def test_matches_torch_multihead_attention_for_cross_attention(self) -> None:
        """Test that the equivalence also holds when the keys are a different sequence."""
        expected, _ = self.attention(self.query, self.key, self.value, need_weights=False)

        attended, _ = self.exportable_attention(self.query, self.key, self.value)

        self.assertEqual(attended.shape, (self.batch_size, self.num_queries, self.embed_dims))
        self.assertTrue(torch.allclose(attended, expected, atol=1e-5))

    def test_returns_none_for_the_attention_weights(self) -> None:
        """
        Test that the module returns the two-tuple nn.MultiheadAttention returns, so a decoder can
        call either one with the same unpacking.
        """
        outputs = self.exportable_attention(self.query, self.key, self.value)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)
        self.assertIsNone(outputs[1])

    def test_copies_the_packed_projection_weights(self) -> None:
        """
        Test that the packed in_proj weights are split into the q, k and v projections in that
        order, since a swap would still run and still produce plausible numbers.
        """
        q_weight, k_weight, v_weight = self.attention.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.attention.in_proj_bias.chunk(3, dim=0)

        self.assertTrue(torch.equal(self.exportable_attention.q_proj.weight, q_weight))
        self.assertTrue(torch.equal(self.exportable_attention.k_proj.weight, k_weight))
        self.assertTrue(torch.equal(self.exportable_attention.v_proj.weight, v_weight))
        self.assertTrue(torch.equal(self.exportable_attention.q_proj.bias, q_bias))
        self.assertTrue(torch.equal(self.exportable_attention.k_proj.bias, k_bias))
        self.assertTrue(torch.equal(self.exportable_attention.v_proj.bias, v_bias))
        self.assertTrue(
            torch.equal(self.exportable_attention.out_proj.weight, self.attention.out_proj.weight)
        )
        self.assertTrue(
            torch.equal(self.exportable_attention.out_proj.bias, self.attention.out_proj.bias)
        )

    def test_keeps_the_head_geometry_of_the_source_attention(self) -> None:
        """Test that the head count and per-head width are carried over unchanged."""
        self.assertEqual(self.exportable_attention.embed_dim, self.embed_dims)
        self.assertEqual(self.exportable_attention.num_heads, self.num_heads)
        self.assertEqual(self.exportable_attention.head_dim, self.embed_dims // self.num_heads)
        self.assertEqual(self.exportable_attention.dropout, self.dropout)

    def test_rejects_a_sequence_first_attention(self) -> None:
        """
        Test that a batch_first=False source is rejected, since the module reads its inputs as
        (batch, sequence, channels) and would otherwise attend over the batch axis.
        """
        with self.assertRaises(ValueError):
            ExportableMultiheadAttention(self._build_attention(batch_first=False))

    def test_is_deterministic_in_eval_mode(self) -> None:
        """Test that dropout is inactive in eval mode, so two calls agree exactly."""
        first, _ = self.exportable_attention(self.query, self.key, self.value)
        second, _ = self.exportable_attention(self.query, self.key, self.value)

        self.assertTrue(torch.equal(first, second))

    def test_applies_dropout_only_while_training(self) -> None:
        """
        Test that attention dropout is applied in train mode and not in eval, matching what
        nn.MultiheadAttention does with its own dropout probability.
        """
        self.exportable_attention.train()
        torch.manual_seed(1)
        first, _ = self.exportable_attention(self.query, self.key, self.value)
        torch.manual_seed(2)
        second, _ = self.exportable_attention(self.query, self.key, self.value)

        self.assertFalse(torch.equal(first, second))

    def test_gradients_reach_the_inputs_and_the_projections(self) -> None:
        """Test that the explicit projections stay differentiable, so the module can be trained."""
        query = self.query.clone().requires_grad_(True)
        key = self.key.clone().requires_grad_(True)
        value = self.value.clone().requires_grad_(True)

        attended, _ = self.exportable_attention(query, key, value)
        attended.sum().backward()

        for tensor in (query, key, value):
            assert tensor.grad is not None
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for parameter in self.exportable_attention.parameters():
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_output_stays_finite_for_large_logits(self) -> None:
        """
        Test that inputs large enough to overflow a naive exponential still produce a finite
        output.

        Note this does not pin the explicit max subtraction in the module: torch.softmax subtracts
        its own max, so removing that line is invisible in eager mode. The subtraction is there for
        the exported graph, which this test cannot reach.
        """
        large_query = self.query * 1e4
        large_key = self.key * 1e4

        attended, _ = self.exportable_attention(large_query, large_key, self.value)

        self.assertTrue(torch.isfinite(attended).all())

    def test_follows_the_device_and_dtype_of_the_source_attention(self) -> None:
        """Test that the copied module lands on the same device and dtype as its source."""
        attention = self._build_attention().to(dtype=torch.float64)

        exportable_attention = ExportableMultiheadAttention(attention)

        self.assertEqual(exportable_attention.q_proj.weight.dtype, torch.float64)
        self.assertEqual(exportable_attention.q_proj.weight.device.type, self.device.type)


if __name__ == "__main__":
    unittest.main()
