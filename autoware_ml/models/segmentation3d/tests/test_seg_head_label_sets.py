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

"""Unit tests for set-valued segmentation targets in the PTv3 decoder head loss."""

import unittest

import torch

from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import build_seg_head
from autoware_ml.models.segmentation3d.heads.ptv3 import PTv3SegDecoderHead, voxel_supervision

_CLASSES = ["car", "pole", "sign", "light"]
_SETS = {"thin_or_sign": ["pole", "sign", "light"], "any": ["car", "pole", "sign", "light"]}


def _head_with_sets() -> PTv3SegDecoderHead:
    """Small segmentation head over the test classes with the test label sets."""
    head = build_seg_head(num_classes=4)
    head.set_membership = PTv3SegDecoderHead.build_set_membership(4, _CLASSES, _SETS)
    return head


class TestSetMembership(unittest.TestCase):
    """Unit tests for PTv3SegDecoderHead.build_set_membership."""

    def test_membership_table_marks_the_classes_of_every_set(self) -> None:
        """Test that row k of the table marks the classes of set k."""
        membership = PTv3SegDecoderHead.build_set_membership(4, _CLASSES, _SETS)

        self.assertIsNotNone(membership)
        self.assertEqual(membership.tolist(), [[False, True, True, True], [True, True, True, True]])

    def test_no_sets_build_no_table(self) -> None:
        """Test that a level without label sets has no membership table."""
        for class_sets in (None, {}):
            with self.subTest(class_sets=class_sets):
                self.assertIsNone(PTv3SegDecoderHead.build_set_membership(4, _CLASSES, class_sets))

    def test_unresolvable_sets_are_rejected(self) -> None:
        """Test that sets without class names or with unknown classes raise."""
        with self.assertRaisesRegex(ValueError, "class_names"):
            PTv3SegDecoderHead.build_set_membership(4, None, _SETS)
        with self.assertRaisesRegex(ValueError, "unknown classes"):
            PTv3SegDecoderHead.build_set_membership(4, _CLASSES, {"bad": ["car", "flag"]})


class TestSetTargetLoss(unittest.TestCase):
    """Unit tests for the loss of PTv3SegDecoderHead on class and label set targets."""

    def test_set_targets_train_the_summed_probability_of_the_set(self) -> None:
        """Test that a set target scores the logsumexp of its classes' log probabilities."""
        torch.manual_seed(0)
        head = _head_with_sets()
        logits = torch.randn(6, 4, requires_grad=True)
        # Two class points, two set points (set 0 and set 1), one ignored, one more class point
        targets = torch.tensor([0, 2, 4, 5, -1, 1])
        # voxel_supervision gives the ignored rows weight zero
        losses = head.loss(logits, targets, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0]))

        log_probs = torch.log_softmax(logits.detach(), dim=1)
        ce_single = -(log_probs[0, 0] + log_probs[1, 2] + log_probs[5, 1])
        set0 = -torch.logsumexp(log_probs[2, [1, 2, 3]], dim=0)
        set1 = -torch.logsumexp(log_probs[3, [0, 1, 2, 3]], dim=0)  # log(1) == 0
        expected_ce = (ce_single + set0 + set1) / 5
        torch.testing.assert_close(losses["loss_ce"], expected_ce)
        self.assertTrue(torch.isclose(set1, torch.tensor(0.0), atol=1e-6))

        # Lovasz only sees the class points: same value as with the set points ignored
        class_only = torch.tensor([0, 2, -1, -1, -1, 1])
        torch.testing.assert_close(losses["loss_lovasz"], head.lovasz(logits, class_only))
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # A point whose set covers every class receives no gradient from the cross entropy
        self.assertTrue(torch.allclose(logits.grad[3], torch.zeros(4), atol=1e-6))

    def test_set_targets_without_sets_configured_are_rejected(self) -> None:
        """Test that a head built without sets rejects targets naming a set."""
        head = build_seg_head(num_classes=4)
        logits = torch.randn(2, 4)

        with self.assertRaisesRegex(ValueError, "class_sets"):
            head.loss(logits, torch.tensor([0, 4]), torch.ones(2))

    def test_plain_targets_are_unchanged_by_the_set_machinery(self) -> None:
        """Test that class targets score the same with and without label sets."""
        torch.manual_seed(1)
        logits = torch.randn(5, 4)
        targets = torch.tensor([0, 1, 2, 3, -1])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        for name, head in (("with_sets", _head_with_sets()), ("plain", build_seg_head(4))):
            with self.subTest(head=name):
                losses = head.loss(logits, targets, weights)
                expected = head.cross_entropy(logits, targets).sum() / 4
                torch.testing.assert_close(losses["loss_ce"], expected)
                torch.testing.assert_close(losses["loss_lovasz"], head.lovasz(logits, targets))

    def test_voxel_reduction_keeps_set_labels_and_weights_rows(self) -> None:
        """Test that voxel supervision keeps set labels and the loss scores them."""
        # Voxel 0: two set-0 points and one class point -> majority is the set (index 4).
        # Voxel 1: class points only. Voxel 2: ignored points only.
        point_voxel_indices = torch.tensor([0, 0, 0, 1, 1, 2])
        segment = torch.tensor([4, 4, 1, 2, 2, -1])
        labels, weights = voxel_supervision(
            point_voxel_indices,
            segment,
            num_voxels=3,
            num_classes=4 + len(_SETS),
            ignore_index=-1,
            sample=False,
            mixed_weight=0.0,
        )
        self.assertEqual(labels.tolist(), [4, 2, -1])
        self.assertEqual(weights.tolist(), [1.0, 1.0, 0.0])

        head = _head_with_sets()
        logits = torch.randn(3, 4)
        losses = head.loss(logits, labels, weights)
        log_probs = torch.log_softmax(logits, dim=1)
        expected = -(torch.logsumexp(log_probs[0, [1, 2, 3]], dim=0) + log_probs[1, 2]) / 2
        torch.testing.assert_close(losses["loss_ce"], expected)
