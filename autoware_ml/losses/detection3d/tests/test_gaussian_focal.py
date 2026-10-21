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

"""Unit tests for the Gaussian focal heatmap loss of the 3D detection heads."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss


class TestGaussianFocalLoss(unittest.TestCase):
    """Check the normalization and the weighting of ``GaussianFocalLoss``."""

    def setUp(self) -> None:
        """Build the loss and a two class heatmap with one peak per sample."""
        torch.manual_seed(0)
        self.loss_fn = GaussianFocalLoss()
        self.prediction = torch.randn((2, 2, 4, 4))
        self.target = torch.zeros_like(self.prediction)
        self.target[0, 0, 1, 1] = 1.0
        self.target[1, 1, 2, 2] = 1.0
        self.target[1, 1, 2, 3] = 0.5

    def test_zero_positive_heatmap_gives_a_finite_positive_loss(self) -> None:
        """A heatmap without any peak is normalized by one, not by zero."""
        prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target = torch.zeros_like(prediction)

        loss = self.loss_fn(prediction, target)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_normalizes_by_the_number_of_peaks(self) -> None:
        """The summed loss is divided by the number of cells at the peak value."""
        total = self.loss_fn(self.prediction, self.target)
        per_sample = torch.stack(
            [self.loss_fn(self.prediction[i : i + 1], self.target[i : i + 1]) for i in range(2)]
        )

        # Every sample holds one peak, so the total is the mean of the per-sample losses.
        self.assertTrue(torch.isclose(total, per_sample.sum() / 2.0))

    def test_zero_weighted_class_drops_its_cells_and_its_peaks(self) -> None:
        """A zero class weight removes the class from the loss and from the peak count."""
        # (batch_size, num_classes, 1, 1) weights dropping the second class everywhere.
        weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])[:, :, None, None]

        weighted = self.loss_fn(self.prediction, self.target, weights=weights)
        first_class_only = self.loss_fn(self.prediction[:, :1], self.target[:, :1])

        self.assertTrue(torch.isclose(weighted, first_class_only))
        self.assertFalse(torch.isclose(weighted, self.loss_fn(self.prediction, self.target)))

    def test_weights_scale_the_loss_once(self) -> None:
        """A fractional weight on every cell scales the numerator and the peak count alike."""
        weights = torch.full((2, 2, 1, 1), 0.5)

        weighted = self.loss_fn(self.prediction, self.target, weights=weights)
        unweighted = self.loss_fn(self.prediction, self.target)

        # Both the loss and the peak count halve, so the normalized loss is unchanged. A
        # second application of the weights would halve it.
        self.assertTrue(torch.isclose(weighted, unweighted))

    def test_input_tensors_are_not_mutated(self) -> None:
        """The loss works on copies of the heatmaps."""
        prediction = self.prediction.clone()
        target = self.target.clone()

        self.loss_fn(self.prediction, self.target)

        self.assertTrue(torch.equal(self.prediction, prediction))
        self.assertTrue(torch.equal(self.target, target))


if __name__ == "__main__":
    unittest.main()
