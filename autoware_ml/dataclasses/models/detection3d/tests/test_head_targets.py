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

"""Unit tests for the 3D detection head target dataclasses."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.models.detection3d.head_targets import (
    CenterHeadTargets,
    TransFusionHeadTargets,
)


class TestCenterHeadTargets(unittest.TestCase):
    """Unit tests for the CenterHead training targets."""

    def setUp(self) -> None:
        """Set up the batch layout: two samples, two classes, up to three boxes on a 4x4 grid."""
        self.batch_size, self.num_classes, self.max_num_boxes = 2, 2, 3

    def _build_fields(self, num_reg_targets: int = 10) -> dict[str, torch.Tensor]:
        """Build a fresh, valid set of target tensors with ``num_reg_targets`` regression channels."""
        return {
            "heatmaps": torch.zeros(self.batch_size, self.num_classes, 4, 4),
            "reg_targets": torch.zeros(self.batch_size, self.max_num_boxes, num_reg_targets),
            "reg_indices": torch.zeros(self.batch_size, self.max_num_boxes, dtype=torch.int64),
            "valid_masks": torch.zeros(self.batch_size, self.max_num_boxes, dtype=torch.bool),
        }

    def test_accepts_targets_with_and_without_velocity(self) -> None:
        """Test that both the 8- and 10-channel regression target layouts are accepted."""
        for num_reg_targets in (8, 10):
            with self.subTest(num_reg_targets=num_reg_targets):
                targets = CenterHeadTargets.model_validate(self._build_fields(num_reg_targets))

                self.assertEqual(targets.reg_targets.shape[-1], num_reg_targets)

    def test_valid_masks_must_be_boolean(self) -> None:
        """Test that an integer mask is rejected, so masking never silently multiplies."""
        fields = self._build_fields()
        fields["valid_masks"] = torch.zeros(self.batch_size, self.max_num_boxes, dtype=torch.int64)

        with self.assertRaises(ValidationError):
            CenterHeadTargets.model_validate(fields)

    def test_reg_indices_must_be_int64(self) -> None:
        """Test that gather indices in another dtype are rejected."""
        fields = self._build_fields()
        fields["reg_indices"] = torch.zeros(self.batch_size, self.max_num_boxes, dtype=torch.int32)

        with self.assertRaises(ValidationError):
            CenterHeadTargets.model_validate(fields)

    def test_is_frozen(self) -> None:
        """Test that targets cannot be mutated after construction."""
        fields = self._build_fields()
        targets = CenterHeadTargets.model_validate(fields)

        with self.assertRaises(ValidationError):
            targets.valid_masks = fields["valid_masks"]  # type: ignore[misc]


class TestTransFusionHeadTargets(unittest.TestCase):
    """Unit tests for the TransFusion assignment targets."""

    def setUp(self) -> None:
        """Set up the layout: two samples, four proposals, three classes and a 10-value code."""
        self.batch_size, self.num_proposals, self.num_classes, self.code_size = 2, 4, 3, 10

    def _build_fields(self) -> dict[str, object]:
        """Build a fresh, valid set of constructor fields with no positives."""
        return {
            "labels": torch.zeros(self.batch_size, self.num_proposals, dtype=torch.int64),
            "label_weights": torch.ones(self.batch_size, self.num_proposals, self.num_classes),
            "bbox_targets": torch.zeros(self.batch_size, self.num_proposals, self.code_size),
            "bbox_weights": torch.zeros(self.batch_size, self.num_proposals, self.code_size),
            "num_pos": 0,
            "matched_iou": 0.0,
            "dense_heatmaps": torch.zeros(self.batch_size, self.num_classes, 8, 8),
            "class_weights": torch.ones(self.batch_size, self.num_classes),
        }

    def test_carries_counts_alongside_the_tensors(self) -> None:
        """Test that the positive count and matched IoU travel with the tensors."""
        fields = self._build_fields()
        fields["num_pos"] = 3
        fields["matched_iou"] = 0.5

        targets = TransFusionHeadTargets.model_validate(fields)

        self.assertEqual(targets.num_pos, 3)
        self.assertEqual(targets.matched_iou, 0.5)
        self.assertEqual(
            tuple(targets.label_weights.shape),
            (self.batch_size, self.num_proposals, self.num_classes),
        )

    def test_labels_must_be_int64(self) -> None:
        """Test that class labels in another integer dtype are rejected."""
        fields = self._build_fields()
        fields["labels"] = torch.zeros(self.batch_size, self.num_proposals, dtype=torch.int32)

        with self.assertRaises(ValidationError):
            TransFusionHeadTargets.model_validate(fields)

    def test_label_weights_are_per_class(self) -> None:
        """Test that per-proposal weights without the class axis are rejected."""
        fields = self._build_fields()
        fields["label_weights"] = torch.ones(self.batch_size, self.num_proposals)

        with self.assertRaises(ValidationError):
            TransFusionHeadTargets.model_validate(fields)

    def test_num_pos_must_be_an_int(self) -> None:
        """Test that the positive count is not coerced from a string."""
        fields = self._build_fields()
        fields["num_pos"] = "3"

        with self.assertRaises(ValidationError):
            TransFusionHeadTargets.model_validate(fields)


if __name__ == "__main__":
    unittest.main()
