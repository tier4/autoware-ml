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

"""Unit tests for QualityFocalLoss."""

from __future__ import annotations

import math
import unittest

from jaxtyping import Float, Int64
import torch
import torch.nn.functional as F

from autoware_ml.losses.detection2d.quality_focal import QualityFocalLoss


def _reference_terms(
    logits: Float[torch.Tensor, "num_queries num_classes"],
    labels: Int64[torch.Tensor, " num_queries"],
    quality: Float[torch.Tensor, " num_queries"],
    beta: float,
) -> Float[torch.Tensor, "num_queries num_classes"]:
    """Every term of the loss by the Generalized Focal Loss formula, element by element."""
    terms = torch.empty_like(logits)
    for row in range(logits.shape[0]):
        for column in range(logits.shape[1]):
            logit = logits[row, column]
            target = quality[row] if labels[row] == column else logits.new_tensor(0.0)
            bce = F.binary_cross_entropy_with_logits(logit, target)
            terms[row, column] = bce * (target - logit.sigmoid()).abs() ** beta
    return terms


class TestQualityFocalLoss(unittest.TestCase):
    def setUp(self) -> None:
        """Build the loss with beta 2 and a batch of four queries over three classes.

        Two queries are positives of classes 0 and 2, one is background by the class count and
        one by a negative label.
        """
        self.loss_fn = QualityFocalLoss(beta=2.0)
        generator = torch.Generator().manual_seed(0)
        self.logits = torch.randn(4, 3, generator=generator) * 3
        self.labels = torch.tensor([0, 3, 2, -1])
        self.quality = torch.rand(4, generator=generator)

    def test_the_loss_of_a_background_score_by_hand(self) -> None:
        """
        Input: one score of probability 0.5 for one class, labeled background (label 1 = the
        class count), average factor 1.
        Expected: a background score is pushed toward 0, scaled by its probability to the power
        beta (2): -log(1 - 0.5) * 0.5 ** 2.
        Check: the loss against that value.
        """
        loss = self.loss_fn(torch.zeros(1, 1), torch.tensor([1]), torch.zeros(1), avg_factor=1.0)

        torch.testing.assert_close(loss, torch.tensor(math.log(2.0) * 0.25))

    def test_a_positive_targets_its_iou(self) -> None:
        """
        Input: one positive of class 0 with an IoU quality of 0.7, scored 0.7, 0.4 and 0.95 for
        its class, the other class far below zero.
        Expected: the class score of a positive is pulled toward its IoU rather than 1, so a
        score equal to the IoU leaves only the cross-entropy of the target itself, and a score
        below or above it costs more, scaled by the gap to the power beta (2).
        Check: the loss at 0.7 against the entropy term times a zero gap, and the losses at 0.4
        and 0.95 against the cross-entropy to 0.7 times the squared gap.
        """
        labels = torch.tensor([0])
        quality = torch.tensor([0.7])

        losses = {}
        for probability in (0.7, 0.4, 0.95):
            with self.subTest(probability=probability):
                logit = math.log(probability / (1.0 - probability))
                losses[probability] = self.loss_fn(
                    torch.tensor([[logit, -30.0]]), labels, quality, avg_factor=1.0
                )
                cross_entropy = -(0.7 * math.log(probability) + 0.3 * math.log(1.0 - probability))
                expected = cross_entropy * abs(0.7 - probability) ** 2

                torch.testing.assert_close(losses[probability], torch.tensor(expected))
        self.assertLess(float(losses[0.7]), 1e-6)

    def test_every_term_follows_the_formula(self) -> None:
        """
        Input: four queries over three classes, two positives of classes 0 and 2 and two
        background queries, one labeled by the class count and one by -1, average factor 3.
        Expected: the term of a positive on its own class targets its quality, every other
        term targets 0, and the sum is divided by the average factor.
        Check: the loss and the gradient of every logit against the element-wise formula, so a
        term read at the wrong query or class does not pass.
        """
        logits = self.logits.clone().requires_grad_()
        reference_logits = self.logits.clone().requires_grad_()

        loss = self.loss_fn(logits, self.labels, self.quality, avg_factor=3.0)
        reference = _reference_terms(reference_logits, self.labels, self.quality, 2.0).sum() / 3.0
        loss.backward()
        reference.backward()

        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(logits.grad, reference_logits.grad)

    def test_weights_drop_their_terms(self) -> None:
        """
        Input: the four queries with weights of one, and with the last class weighted zero.
        Expected: weights of one change nothing, and a zero-weighted column leaves the loss of
        the other columns: the loss over the first two classes, where the positive of class 2
        becomes background.
        Check: both weighted losses against the unweighted and the two-class losses.
        """
        weights = torch.ones(4, 3)
        weights[:, 2] = 0.0

        torch.testing.assert_close(
            self.loss_fn(self.logits, self.labels, self.quality, 1.0, weights=torch.ones(4, 3)),
            self.loss_fn(self.logits, self.labels, self.quality, 1.0),
        )
        two_class_labels = torch.where(self.labels >= 2, 2, self.labels)
        torch.testing.assert_close(
            self.loss_fn(self.logits, self.labels, self.quality, 1.0, weights=weights),
            self.loss_fn(self.logits[:, :2], two_class_labels, self.quality, 1.0),
        )

    def test_avg_factor_divides_as_given(self) -> None:
        """
        Input: the four queries scored with an average factor of 4 and of 0.5.
        Expected: the summed loss is divided by the average factor as given, also below 1.
        Check: both losses against the loss with an average factor of 1.
        """
        summed = self.loss_fn(self.logits, self.labels, self.quality, avg_factor=1.0)

        torch.testing.assert_close(
            self.loss_fn(self.logits, self.labels, self.quality, avg_factor=4.0), summed / 4.0
        )
        torch.testing.assert_close(
            self.loss_fn(self.logits, self.labels, self.quality, avg_factor=0.5), summed * 2.0
        )

    def test_half_precision_inputs_are_refused(self) -> None:
        """
        Input: the logits, then the qualities, in bfloat16 and in float16.
        Expected: in bfloat16 a positive scored within 1/256 of its quality gets no gradient,
        so the loss takes float32 only and a head training under autocast casts first.
        Check: the ValueError naming float32 and the tensor.
        """
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "float32 logits"):
                    self.loss_fn(self.logits.to(dtype), self.labels, self.quality, avg_factor=1.0)
                with self.assertRaisesRegex(ValueError, "float32 quality_scores"):
                    self.loss_fn(self.logits, self.labels, self.quality.to(dtype), avg_factor=1.0)


if __name__ == "__main__":
    unittest.main()
