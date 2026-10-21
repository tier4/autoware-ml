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

"""Unit tests for the sigmoid focal loss of the 3D detection heads."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss


class TestSigmoidFocalLoss(unittest.TestCase):
    """Check the normalization and the weighting of ``SigmoidFocalLoss``."""

    def setUp(self) -> None:
        """Build the loss and two one-hot queries."""
        self.loss_fn = SigmoidFocalLoss()
        self.logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
        self.targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    def test_matches_the_focal_formula_on_one_query(self) -> None:
        """The loss of one query is the alpha balanced, gamma focused cross entropy."""
        loss_fn = SigmoidFocalLoss(gamma=2.0, alpha=0.25)
        logits = torch.tensor([[1.0]])
        targets = torch.tensor([[1.0]])
        probability = torch.sigmoid(logits)
        expected = -0.25 * (1.0 - probability).pow(2.0) * torch.log(probability)

        loss = loss_fn(logits, targets)

        self.assertTrue(torch.isclose(loss, expected.sum()))

    def test_sums_over_queries_and_classes_without_avg_factor(self) -> None:
        """Without a normalization factor the loss is the plain sum over the queries."""
        loss = self.loss_fn(self.logits, self.targets)
        per_query = torch.stack(
            [self.loss_fn(self.logits[i : i + 1], self.targets[i : i + 1]) for i in range(2)]
        )

        self.assertTrue(torch.isclose(loss, per_query.sum()))

    def test_avg_factor_divides_the_sum(self) -> None:
        """The normalization factor divides the summed loss."""
        unnormalized = self.loss_fn(self.logits, self.targets)

        normalized = self.loss_fn(self.logits, self.targets, avg_factor=4.0)

        self.assertTrue(torch.isclose(normalized, unnormalized / 4.0))

    def test_avg_factor_is_clamped_to_one(self) -> None:
        """A normalization factor below one does not inflate the loss."""
        unnormalized = self.loss_fn(self.logits, self.targets)

        clamped = self.loss_fn(self.logits, self.targets, avg_factor=0.0)

        self.assertTrue(torch.isclose(clamped, unnormalized))

    def test_zero_weighted_query_is_dropped(self) -> None:
        """A query whose weights are zero contributes nothing to the loss."""
        weights = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

        weighted = self.loss_fn(self.logits, self.targets, weights=weights)
        first_query_only = self.loss_fn(self.logits[:1], self.targets[:1])

        self.assertTrue(torch.isclose(weighted, first_query_only))

    def test_zero_weighted_class_is_dropped(self) -> None:
        """A zero per-class weight drops that class of every query."""
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-3.0, 1.5]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        weights = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        weighted = self.loss_fn(logits, targets, weights=weights)
        first_class_only = self.loss_fn(logits[:, :1], targets[:, :1])

        self.assertTrue(torch.isclose(weighted, first_class_only))

    def test_weights_scale_the_loss_linearly(self) -> None:
        """A fractional weight scales the loss of its query by that weight, once."""
        weights = torch.full_like(self.logits, 0.5)

        weighted = self.loss_fn(self.logits, self.targets, weights=weights)
        unweighted = self.loss_fn(self.logits, self.targets)

        self.assertTrue(torch.isclose(weighted, unweighted * 0.5))


if __name__ == "__main__":
    unittest.main()
