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

"""Tests for the point-wise Lovasz-Softmax loss.

The expected values come from the definition of the loss (Berman et al., 2018):
for every class present in the labels, sort the errors ``|1[y = c] - p_c|`` in
descending order and take their dot product with the discrete gradient of the
Jaccard loss along that order; the loss is the mean over the present classes.
"""

import math

import pytest
import torch

from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss

IGNORE = -1


def _logits_for(probabilities: torch.Tensor) -> torch.Tensor:
    """Logits whose softmax is exactly ``probabilities`` (zero entries allowed)."""
    return torch.where(probabilities > 0, probabilities.log(), torch.full_like(probabilities, -1e4))


def test_hand_computed_two_class_example() -> None:
    """Three points, two classes, worked through by hand from the definition.

    Probabilities for class 0 are 0.9, 0.6, 0.2; labels are 0, 0, 1. For each
    present class: take the errors ``|1[y = c] - p_c|``, sort them descending,
    and after each prefix of the sorted order compute the Jaccard loss
    ``1 - intersection / union`` between the prefix and the ground truth. The
    class term is the dot product of the sorted errors with the increments of
    that Jaccard loss.

    Class 0 (ground truth: points 0 and 1)::

        point               0     1     2
        error               0.1   0.4   0.2
        sorted (point)      1     2     0
        sorted error        0.4   0.2   0.1
        is foreground       1     0     1
        intersection        1     1     0
        union               2     3     3
        Jaccard loss        1/2   2/3   1
        increment           1/2   1/6   1/3

        term = 0.4 * 1/2 + 0.2 * 1/6 + 0.1 * 1/3 = 4/15

    Class 1 (ground truth: point 2; p_1 = 0.1, 0.4, 0.8)::

        sorted (point)      1     2     0
        sorted error        0.4   0.2   0.1
        is foreground       0     1     0
        intersection        1     0     0
        union               2     2     3
        Jaccard loss        1/2   1     1
        increment           1/2   1/2   0

        term = 0.4 * 1/2 + 0.2 * 1/2 + 0.1 * 0 = 3/10

    The loss is the mean of the two terms.
    """
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1])
    loss = LovaszLoss(ignore_index=IGNORE)(_logits_for(probabilities), labels)
    assert math.isclose(loss.item(), (4 / 15 + 3 / 10) / 2, rel_tol=1e-9)


def test_perfect_prediction_has_zero_loss() -> None:
    labels = torch.tensor([0, 2, 1, 2, 0])
    probabilities = torch.nn.functional.one_hot(labels, 3).to(torch.float64)
    loss = LovaszLoss(ignore_index=IGNORE)(_logits_for(probabilities), labels)
    assert loss.item() == 0.0


def test_completely_wrong_prediction_has_unit_loss() -> None:
    """With zero mass on the true class of every point, each present class has
    error 1 on all its points, so its Lovasz term is the full Jaccard loss, 1."""
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    probabilities = torch.nn.functional.one_hot((labels + 1) % 3, 3).to(torch.float64)
    loss = LovaszLoss(ignore_index=IGNORE)(_logits_for(probabilities), labels)
    assert math.isclose(loss.item(), 1.0, rel_tol=1e-9)


def test_loss_is_bounded_by_zero_and_one() -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        logits = torch.randn(200, 7, generator=generator) * 3
        labels = torch.randint(0, 7, (200,), generator=generator)
        loss = LovaszLoss(ignore_index=IGNORE)(logits, labels)
        assert 0.0 <= loss.item() <= 1.0


def test_absent_classes_do_not_enter_the_mean() -> None:
    """A class with no points is not averaged in: the two-class example must
    give the same loss when a third, never-labelled class is appended with zero
    probability, so the present classes' probabilities are unchanged."""
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1])
    two_class = LovaszLoss(ignore_index=IGNORE)(_logits_for(probabilities), labels)
    padded = torch.cat([probabilities, torch.zeros(3, 1, dtype=torch.float64)], dim=1)
    three_class = LovaszLoss(ignore_index=IGNORE)(_logits_for(padded), labels)
    assert math.isclose(two_class.item(), three_class.item(), rel_tol=1e-9)
    assert math.isclose(two_class.item(), (4 / 15 + 0.3) / 2, rel_tol=1e-9)


def test_ignored_points_do_not_affect_the_loss_or_receive_gradient() -> None:
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(120, 5, generator=generator)
    labels = torch.randint(0, 5, (120,), generator=generator)
    ignored = torch.rand(120, generator=generator) < 0.3
    labels_with_ignore = labels.clone()
    labels_with_ignore[ignored] = IGNORE
    loss_fn = LovaszLoss(ignore_index=IGNORE)

    full = logits.clone().requires_grad_(True)
    loss = loss_fn(full, labels_with_ignore)
    (grad,) = torch.autograd.grad(loss, full)
    reference = loss_fn(logits[~ignored], labels[~ignored])

    assert torch.allclose(loss, reference)
    assert torch.all(grad[ignored] == 0)
    assert torch.any(grad[~ignored] != 0)


def test_loss_is_invariant_to_the_order_of_points() -> None:
    generator = torch.Generator().manual_seed(2)
    logits = torch.randn(80, 4, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 4, (80,), generator=generator)
    permutation = torch.randperm(80, generator=generator)
    loss_fn = LovaszLoss(ignore_index=IGNORE)
    shuffled = loss_fn(logits[permutation], labels[permutation])
    assert torch.allclose(loss_fn(logits, labels), shuffled)


def test_gradient_matches_finite_differences() -> None:
    """The loss is piecewise linear in the probabilities; away from ties in the
    sort its autograd gradient must agree with central finite differences."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(12, 3, generator=generator, dtype=torch.float64).requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2])
    loss_fn = LovaszLoss(ignore_index=IGNORE)
    assert torch.autograd.gradcheck(lambda x: loss_fn(x, labels), (logits,), eps=1e-6, atol=1e-5)


def test_all_ignored_batch_keeps_the_graph() -> None:
    """An all-ignored batch yields a zero loss that still differentiates to zeros."""
    logits = torch.randn(32, 15, requires_grad=True)
    loss = LovaszLoss(ignore_index=IGNORE)(logits, torch.full((32,), IGNORE))
    (grad,) = torch.autograd.grad(loss, logits)
    assert loss.item() == 0.0
    assert torch.all(grad == 0)


@pytest.mark.parametrize("weight", [1.0, 0.5])
def test_per_image_averages_the_per_sample_losses(weight: float) -> None:
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(2, 5, 100, generator=generator)  # (batch, classes, points)
    labels = torch.randint(0, 5, (2, 100), generator=generator)
    labels[0, :10] = IGNORE
    joint = LovaszLoss(ignore_index=IGNORE)
    per_sample = torch.stack([joint(logits[i].T, labels[i]) for i in range(2)])
    per_image = LovaszLoss(ignore_index=IGNORE, per_image=True, loss_weight=weight)(logits, labels)
    assert torch.allclose(per_image, weight * per_sample.mean())
