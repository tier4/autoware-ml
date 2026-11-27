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

"""Unit tests for the tensor functions shared by the models of more than one task."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.functional import build_denoising_attention_mask, inverse_sigmoid


class TestInverseSigmoid(unittest.TestCase):
    """Check the clamped logit used to re-encode reference points."""

    def test_inverse_sigmoid_inverts_sigmoid(self) -> None:
        """Test that a sigmoid after inverse_sigmoid recovers values inside (0, 1).

        Input: 0.2, 0.5 and 0.9.
        Expected: the same values after ``inverse_sigmoid`` and then ``sigmoid``.
        Check: the round trip against the input.
        """
        values = torch.tensor([0.2, 0.5, 0.9])

        torch.testing.assert_close(inverse_sigmoid(values).sigmoid(), values, rtol=1e-5, atol=1e-6)

    def test_inverse_sigmoid_clamps_extremes_to_finite_values(self) -> None:
        """Test that values at or beyond the (0, 1) boundaries stay finite.

        Input: 0, 1, -1 and 2.
        Expected: a finite result for each.
        Check: every output is finite.
        """
        self.assertTrue(torch.isfinite(inverse_sigmoid(torch.tensor([0.0, 1.0, -1.0, 2.0]))).all())


class TestBuildDenoisingAttentionMask(unittest.TestCase):
    """Check which queries the denoising attention mask lets see each other."""

    def test_groups_see_themselves_and_the_matching_queries(self) -> None:
        """Test that a group sees itself and the matching queries, and matching only itself.

        Input: 3 groups of 2 denoising queries before 4 matching queries.
        Expected: every group sees its own 2 queries and the 4 matching ones; a matching query
        sees the 4 matching ones only.
        Check: the mask against the expected visible blocks; True blocks attention.
        """
        mask = build_denoising_attention_mask(4, 3, 2, torch.device("cpu"))

        visible = torch.zeros(10, 10, dtype=torch.bool)
        for group in range(3):
            rows = slice(2 * group, 2 * (group + 1))
            visible[rows, rows] = True
        visible[:, 6:] = True
        torch.testing.assert_close(mask, ~visible)

    def test_denoising_queries_can_follow_the_matching_queries(self) -> None:
        """Test that the same rule holds with the denoising queries after the matching ones.

        Input: 4 matching queries before 2 groups of 3 denoising queries.
        Expected: a matching query sees the 4 matching ones only; every group sees its own 3
        queries and the 4 matching ones.
        Check: the mask against the expected visible blocks; True blocks attention.
        """
        mask = build_denoising_attention_mask(4, 2, 3, torch.device("cpu"), denoising_first=False)

        visible = torch.zeros(10, 10, dtype=torch.bool)
        visible[:, :4] = True
        visible[4:7, 4:7] = True
        visible[7:, 7:] = True
        torch.testing.assert_close(mask, ~visible)

    def test_mask_equals_the_group_by_group_construction(self) -> None:
        """Test that the mask equals the one built group by group, as the detectors did.

        Input: 1 to 5 groups of 1 to 4 denoising queries before 1 and 5 matching queries.
        Expected: the mask a loop over the groups builds: matching rows block every denoising
        column, and every group blocks the groups before and after it.
        Check: both masks for each of the 40 layouts.
        """
        for num_groups in range(1, 6):
            for group_size in range(1, 5):
                for num_matching_queries in (1, 5):
                    num_denoising = num_groups * group_size
                    num_total = num_denoising + num_matching_queries
                    expected = torch.zeros(num_total, num_total, dtype=torch.bool)
                    expected[num_denoising:, :num_denoising] = True
                    for group in range(num_groups):
                        rows = slice(group_size * group, group_size * (group + 1))
                        expected[rows, : group_size * group] = True
                        expected[rows, group_size * (group + 1) : num_denoising] = True

                    mask = build_denoising_attention_mask(
                        num_matching_queries, num_groups, group_size, torch.device("cpu")
                    )

                    torch.testing.assert_close(mask, expected)

    def test_no_denoising_queries_blocks_nothing(self) -> None:
        """Test that a query set without denoising queries blocks nothing.

        Input: 3 matching queries and one group of 0 denoising queries.
        Expected: a 3 x 3 mask without any blocked pair.
        Check: the shape and that no entry is True.
        """
        mask = build_denoising_attention_mask(3, 1, 0, torch.device("cpu"))

        self.assertEqual(tuple(mask.shape), (3, 3))
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
