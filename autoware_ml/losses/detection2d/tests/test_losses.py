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

"""Unit tests for the losses of the auxiliary 2D head."""

from __future__ import annotations

import math

import torch
from torchvision.ops import generalized_box_iou

from autoware_ml.losses.detection2d.losses import GIoULoss, QualityFocalLoss, WeightedL1Loss


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def test_quality_focal_loss_of_a_background_score_by_hand() -> None:
    """
    Input: one score of probability 0.5 for one class, labeled background (label 1 = the class
    count).
    Expected: a background score is pushed toward 0, scaled by its probability to the power
    beta (2): -log(1 - 0.5) * 0.5 ** 2.
    Check: the loss against that value.
    """
    loss = QualityFocalLoss(beta=2.0)(torch.zeros(1, 1), torch.tensor([1]), torch.zeros(1))

    torch.testing.assert_close(loss, torch.tensor(math.log(2.0) * 0.25))


def test_quality_focal_loss_targets_the_iou_of_a_positive() -> None:
    """
    Input: one positive of class 0 with an IoU quality of 0.7, scored 0.7, 0.4 and 0.95 for its
    class, the other class far below zero.
    Expected: the class score of a positive is pulled toward its IoU rather than 1, so a score
    equal to the IoU leaves only the cross-entropy of the target itself, and a score below or
    above it costs more, scaled by the gap to the power beta (2).
    Check: the loss at 0.7 against the entropy term times a zero gap, and the losses at 0.4 and
    0.95 against the cross-entropy to 0.7 times the squared gap.
    """
    loss_fn = QualityFocalLoss(beta=2.0)
    labels = torch.tensor([0])
    quality = torch.tensor([0.7])

    def loss_at(probability: float) -> torch.Tensor:
        return loss_fn(torch.tensor([[_logit(probability), -30.0]]), labels, quality)

    def expected(probability: float) -> float:
        cross_entropy = -(0.7 * math.log(probability) + 0.3 * math.log(1.0 - probability))
        return cross_entropy * abs(0.7 - probability) ** 2

    for probability in (0.7, 0.4, 0.95):
        torch.testing.assert_close(loss_at(probability), torch.tensor(expected(probability)))
    assert float(loss_at(0.7)) < 1e-6


def test_quality_focal_loss_class_weights_drop_their_columns() -> None:
    """
    Input: random scores of four rows over three classes, one positive per row, with class
    weights of one and with the last column weighted zero.
    Expected: weights of one change nothing, and a zero-weighted column leaves the loss of the
    other columns: the loss over the first two classes, where the positives of class 2 become
    background.
    Check: both weighted losses against the unweighted and the two-class losses.
    """
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 3, generator=generator)
    labels = torch.tensor([0, 1, 2, 3])
    quality = torch.rand(4, generator=generator)
    loss_fn = QualityFocalLoss()
    weights = torch.ones(4, 3)
    weights[:, 2] = 0.0

    torch.testing.assert_close(
        loss_fn(logits, labels, quality, class_weights=torch.ones(4, 3)),
        loss_fn(logits, labels, quality),
    )
    two_class_labels = torch.where(labels >= 2, 2, labels)
    torch.testing.assert_close(
        loss_fn(logits, labels, quality, class_weights=weights),
        loss_fn(logits[:, :2], two_class_labels, quality),
    )


def test_quality_focal_loss_normalization_and_weight() -> None:
    """
    Input: random scores of four rows over three classes, scored with an average factor of 4
    and a loss weight of 2, and with an average factor of 0.5.
    Expected: the summed loss is divided by the average factor as given, also below 1, then
    multiplied by the loss weight.
    Check: both losses against the plain summed loss.
    """
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 3, generator=generator)
    labels = torch.tensor([0, 3, 1, 3])
    quality = torch.rand(4, generator=generator)
    summed = QualityFocalLoss()(logits, labels, quality)

    torch.testing.assert_close(
        QualityFocalLoss(loss_weight=2.0)(logits, labels, quality, avg_factor=4.0), summed / 2.0
    )
    torch.testing.assert_close(
        QualityFocalLoss()(logits, labels, quality, avg_factor=0.5), summed * 2.0
    )


def _random_boxes(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    corner = torch.rand(count, 2, generator=generator) * 100.0
    size = torch.rand(count, 2, generator=generator) * 40.0 + 1.0
    return torch.cat([corner, corner + size], dim=-1)


def test_giou_loss_matches_torchvision() -> None:
    """
    Input: eight random predicted and target box pairs, scored with an average factor of 4 and
    a loss weight of 2, and every box against itself.
    Expected: the loss of a pair is 1 - GIoU, summed over the pairs, divided by the average
    factor and multiplied by the loss weight; a box matching its target costs nothing.
    Check: the loss against 1 - torchvision's GIoU of the pairs, and the loss of identical
    boxes against zero.
    """
    pred = _random_boxes(8, seed=0)
    target = _random_boxes(8, seed=1)
    expected = (1.0 - generalized_box_iou(pred, target).diagonal()).sum() * 2.0 / 4.0

    torch.testing.assert_close(GIoULoss(loss_weight=2.0)(pred, target, avg_factor=4.0), expected)
    torch.testing.assert_close(GIoULoss()(pred, pred), torch.tensor(0.0))


def test_giou_loss_averages_per_coordinate_weights_per_box() -> None:
    """
    Input: three box pairs with per-coordinate weights (one row per box), the second box
    weighted zero and the third half weighted on two of its coordinates.
    Expected: like the reference, per-coordinate weights are averaged into one weight per box,
    so the loss is the per-box GIoU loss weighted 1, 0 and 0.75.
    Check: the loss against the weighted sum of the per-box losses.
    """
    pred = _random_boxes(3, seed=2)
    target = _random_boxes(3, seed=3)
    weights = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5]])
    per_box = 1.0 - generalized_box_iou(pred, target).diagonal()

    torch.testing.assert_close(
        GIoULoss()(pred, target, weights), (per_box * torch.tensor([1.0, 0.0, 0.75])).sum()
    )


def test_weighted_l1_loss_by_hand() -> None:
    """
    Input: predictions (1, 2, 3) against targets (0, 0, 0), weighted (1, 0, 2), with an average
    factor of 2 and a loss weight of 3.
    Expected: the weighted absolute errors sum to 1 + 0 + 6 = 7, divided by 2 and multiplied by
    3.
    Check: the loss against 10.5.
    """
    loss = WeightedL1Loss(loss_weight=3.0)(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.zeros(3),
        torch.tensor([1.0, 0.0, 2.0]),
        avg_factor=2.0,
    )

    torch.testing.assert_close(loss, torch.tensor(10.5))
