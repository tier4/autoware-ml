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

from autoware_ml.models.common.functional import inverse_sigmoid


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


if __name__ == "__main__":
    unittest.main()
