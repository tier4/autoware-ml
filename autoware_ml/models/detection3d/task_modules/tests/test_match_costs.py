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

"""Unit tests for the Hungarian matching costs."""

import unittest
from typing import Sequence

from jaxtyping import Bool, Float32, Int64
import torch

from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)


class TestClassificationCost(unittest.TestCase):
    """Unit tests for the ClassificationCost function."""

    def setUp(self) -> None:
        """Set up the same logits, labels and masks for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 2
        self.num_predictions = 4
        self.num_classes = 3
        self.max_num_gt_bboxes = 3
        self.cls_cost = ClassificationCost(weight=0.15, alpha=0.25, gamma=2.0)

        # (batch_size, num_predictions, num_classes)
        self.cls_logits = torch.tensor(
            [
                [
                    [4.0, -3.0, -3.0],
                    [-3.0, 4.0, -3.0],
                    [-3.0, -3.0, 4.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [2.0, 1.0, -1.0],
                    [-1.0, 2.0, 1.0],
                    [1.0, -1.0, 2.0],
                    [-2.0, -2.0, -2.0],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        # (batch_size, max_num_gt_bboxes). The batched pipeline pads labels with -1
        # (see Detection3DGTBatch), so the padded entries carry that value here too.
        self.gt_labels = torch.tensor(
            [[0, 2, -1], [1, -1, -1]], dtype=torch.int64, device=self.device
        )
        # (batch_size, max_num_gt_bboxes)
        self.valid_masks = torch.tensor(
            [[True, True, False], [True, False, False]], device=self.device
        )

    def _expected_classification_cost(
        self,
        cls_logits: Float32[torch.Tensor, "batch_size num_predictions num_classes"],
        gt_labels: Int64[torch.Tensor, "batch_size max_num_gt_bboxes"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_predictions max_num_gt_bboxes"]:
        """Build the reference cost with an explicit per-sample, per-pair Python loop."""
        batch_size, num_predictions, _ = cls_logits.shape
        max_num_gt_bboxes = gt_labels.shape[1]
        alpha, gamma = self.cls_cost.alpha, self.cls_cost.gamma
        eps = self.cls_cost.eps

        cost = torch.zeros(
            (batch_size, num_predictions, max_num_gt_bboxes),
            dtype=torch.float32,
            device=cls_logits.device,
        )
        for batch_index in range(batch_size):
            for prediction_index in range(num_predictions):
                for gt_index in range(max_num_gt_bboxes):
                    if not bool(valid_masks[batch_index, gt_index]):
                        cost[batch_index, prediction_index, gt_index] = torch.finfo(
                            torch.float32
                        ).max
                        continue
                    class_id = int(gt_labels[batch_index, gt_index])
                    prob = float(
                        cls_logits[batch_index, prediction_index, class_id].sigmoid().item()
                    )
                    prob = min(max(prob, eps), 1.0 - eps)
                    pos_cost = -torch.tensor(prob + eps).log() * alpha * (1.0 - prob) ** gamma
                    neg_cost = -torch.tensor(1.0 - prob + eps).log() * (1.0 - alpha) * prob**gamma
                    cost[batch_index, prediction_index, gt_index] = (
                        pos_cost - neg_cost
                    ) * self.cls_cost.weight
        return cost

    def test_classification_cost_matches_per_sample_reference(self) -> None:
        """Test that the batched cost matches an explicit per-pair reference loop."""
        cost = self.cls_cost(self.cls_logits, self.gt_labels, self.valid_masks)

        self.assertEqual(
            cost.shape, (self.batch_size, self.num_predictions, self.max_num_gt_bboxes)
        )
        expected_cost = self._expected_classification_cost(
            self.cls_logits, self.gt_labels, self.valid_masks
        )
        self.assertTrue(torch.allclose(cost, expected_cost, atol=1e-6))

    def test_classification_cost_is_independent_across_batch(self) -> None:
        """Test that each sample's cost block only depends on that sample's inputs."""
        batched_cost = self.cls_cost(self.cls_logits, self.gt_labels, self.valid_masks)

        for batch_index in range(self.batch_size):
            single_cost = self.cls_cost(
                self.cls_logits[batch_index : batch_index + 1],
                self.gt_labels[batch_index : batch_index + 1],
                self.valid_masks[batch_index : batch_index + 1],
            )
            self.assertTrue(
                torch.allclose(single_cost[0], batched_cost[batch_index], atol=1e-6),
                msg=f"batch element {batch_index} depends on the rest of the batch",
            )

    def test_classification_cost_gathers_the_gt_class_channel(self) -> None:
        """Test that the cost reads the gt's own class channel, not the prediction axis."""
        cost = self.cls_cost(self.cls_logits, self.gt_labels, self.valid_masks)

        # Sample 0 gt 0 is class 0 and gt 1 is class 2. Prediction 0 is confident about class 0
        # and prediction 2 about class 2, so each must be the cheapest match for its own gt.
        self.assertEqual(int(cost[0, :, 0].argmin()), 0)
        self.assertEqual(int(cost[0, :, 1].argmin()), 2)
        # Sample 1 gt 0 is class 1, which prediction 1 is the most confident about.
        self.assertEqual(int(cost[1, :, 0].argmin()), 1)
        # A confident correct prediction earns a negative cost, since the cost is the marginal
        # focal loss of flipping that class channel's target from 0 to 1.
        self.assertLess(float(cost[0, 0, 0]), 0.0)

    def test_classification_cost_masks_invalid_gt_bboxes(self) -> None:
        """Test that padded gt columns are filled with the highest possible cost."""
        cost = self.cls_cost(self.cls_logits, self.gt_labels, self.valid_masks)

        # (batch_size, 1, max_num_gt_bboxes) broadcast over the prediction axis
        invalid_columns = ~self.valid_masks.unsqueeze(1).expand_as(cost)
        self.assertTrue(torch.all(cost[invalid_columns] == torch.finfo(cost.dtype).max))
        self.assertTrue(torch.all(cost[~invalid_columns] < torch.finfo(cost.dtype).max))

    def test_classification_cost_ignores_padded_labels(self) -> None:
        """Test that the -1 padding of invalid gt labels neither raises nor changes the cost."""
        padded_gt_labels = self.gt_labels.clone()
        # Out-of-range padding in both directions would make a raw gather raise.
        padded_gt_labels[0, 2] = -1
        padded_gt_labels[1, 1] = -7
        padded_gt_labels[1, 2] = self.num_classes + 5

        cost = self.cls_cost(self.cls_logits, padded_gt_labels, self.valid_masks)
        expected_cost = self.cls_cost(self.cls_logits, self.gt_labels, self.valid_masks)
        self.assertTrue(torch.allclose(cost, expected_cost, atol=1e-6))

    def test_classification_cost_scales_with_weight(self) -> None:
        """Test that the weight scales only the valid entries of the cost."""
        unit_cost = ClassificationCost(weight=1.0)(
            self.cls_logits, self.gt_labels, self.valid_masks
        )
        scaled_cost = ClassificationCost(weight=0.15)(
            self.cls_logits, self.gt_labels, self.valid_masks
        )

        valid_columns = self.valid_masks.unsqueeze(1).expand_as(unit_cost)
        self.assertTrue(
            torch.allclose(scaled_cost[valid_columns], unit_cost[valid_columns] * 0.15, atol=1e-6)
        )


class TestBBoxBEVL1Cost(unittest.TestCase):
    """Unit tests for the BBoxBEVL1Cost function."""

    def setUp(self) -> None:
        """Set up the same centers, masks and detector range for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 2
        self.num_predictions = 4
        self.max_num_gt_bboxes = 3
        self.reg_cost = BBoxBEVL1Cost(weight=0.25)
        self.point_cloud_range = [-40.0, -40.0, -3.0, 40.0, 40.0, 5.0]

        # (batch_size, num_predictions, 2). More than two predictions per sample, so a cost that
        # sliced the prediction axis instead of the coordinate axis would change the shape.
        self.bboxes_centers = torch.tensor(
            [
                [[0.0, 0.0], [10.0, -10.0], [-20.0, 20.0], [39.0, 39.0]],
                [[1.0, 2.0], [-30.0, 5.0], [25.0, -25.0], [0.0, 35.0]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        # (batch_size, max_num_gt_bboxes, 2)
        self.gt_bboxes_centers = torch.tensor(
            [
                [[0.5, -0.5], [-20.0, 20.0], [7.0, 7.0]],
                [[24.0, -24.0], [-8.0, -8.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        # (batch_size, max_num_gt_bboxes)
        self.valid_masks = torch.tensor(
            [[True, True, False], [True, True, True]], device=self.device
        )

    def _expected_bbox_bev_l1_cost(
        self,
        bboxes_centers: Float32[torch.Tensor, "batch_size num_predictions 2"],
        gt_bboxes_centers: Float32[torch.Tensor, "batch_size max_num_gt_bboxes 2"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
        point_cloud_range: Sequence[float],
    ) -> Float32[torch.Tensor, "batch_size num_predictions max_num_gt_bboxes"]:
        """Build the reference cost with an explicit per-sample, per-pair Python loop."""
        batch_size, num_predictions, _ = bboxes_centers.shape
        max_num_gt_bboxes = gt_bboxes_centers.shape[1]
        pc_start = torch.tensor(point_cloud_range[0:2], device=bboxes_centers.device)
        pc_extent = torch.tensor(point_cloud_range[3:5], device=bboxes_centers.device) - pc_start

        cost = torch.zeros(
            (batch_size, num_predictions, max_num_gt_bboxes),
            dtype=torch.float32,
            device=bboxes_centers.device,
        )
        for batch_index in range(batch_size):
            for prediction_index in range(num_predictions):
                for gt_index in range(max_num_gt_bboxes):
                    if not bool(valid_masks[batch_index, gt_index]):
                        cost[batch_index, prediction_index, gt_index] = torch.finfo(
                            torch.float32
                        ).max
                        continue
                    norm_bbox = (
                        bboxes_centers[batch_index, prediction_index] - pc_start
                    ) / pc_extent
                    norm_gt_bbox = (gt_bboxes_centers[batch_index, gt_index] - pc_start) / pc_extent
                    cost[batch_index, prediction_index, gt_index] = (
                        norm_bbox - norm_gt_bbox
                    ).abs().sum() * self.reg_cost.weight
        return cost

    def test_bbox_bev_l1_cost_matches_per_sample_reference(self) -> None:
        """Test that the batched cost matches an explicit per-pair reference loop."""
        cost = self.reg_cost(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )

        self.assertEqual(
            cost.shape, (self.batch_size, self.num_predictions, self.max_num_gt_bboxes)
        )
        expected_cost = self._expected_bbox_bev_l1_cost(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )
        self.assertTrue(torch.allclose(cost, expected_cost, atol=1e-6))

    def test_bbox_bev_l1_cost_is_independent_across_batch(self) -> None:
        """Test that each sample's cost block only depends on that sample's inputs."""
        batched_cost = self.reg_cost(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )

        for batch_index in range(self.batch_size):
            single_cost = self.reg_cost(
                self.bboxes_centers[batch_index : batch_index + 1],
                self.gt_bboxes_centers[batch_index : batch_index + 1],
                self.valid_masks[batch_index : batch_index + 1],
                self.point_cloud_range,
            )
            self.assertTrue(
                torch.allclose(single_cost[0], batched_cost[batch_index], atol=1e-6),
                msg=f"batch element {batch_index} depends on the rest of the batch",
            )

    def test_bbox_bev_l1_cost_prefers_the_nearest_center(self) -> None:
        """Test that the cheapest prediction for each gt is the closest one in BEV."""
        cost = self.reg_cost(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )

        # Sample 0 gt 0 is at (0.5, -0.5), closest to prediction 0 at (0, 0). Its gt 1 sits exactly
        # on prediction 2, so that pair must cost zero.
        self.assertEqual(int(cost[0, :, 0].argmin()), 0)
        self.assertEqual(int(cost[0, :, 1].argmin()), 2)
        self.assertAlmostEqual(float(cost[0, 2, 1]), 0.0, places=6)
        # Sample 1 gt 0 is at (24, -24), closest to prediction 2 at (25, -25).
        self.assertEqual(int(cost[1, :, 0].argmin()), 2)

    def test_bbox_bev_l1_cost_masks_invalid_gt_bboxes(self) -> None:
        """Test that padded gt columns are filled with the highest possible cost."""
        cost = self.reg_cost(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )

        invalid_columns = ~self.valid_masks.unsqueeze(1).expand_as(cost)
        self.assertTrue(torch.all(cost[invalid_columns] == torch.finfo(cost.dtype).max))
        self.assertTrue(torch.all(cost[~invalid_columns] < torch.finfo(cost.dtype).max))

    def test_bbox_bev_l1_cost_ignores_padded_centers(self) -> None:
        """Test that the geometry of invalid gt boxes cannot change the valid costs."""
        padded_gt_bboxes_centers = self.gt_bboxes_centers.clone()
        padded_gt_bboxes_centers[0, 2] = torch.tensor([1e6, -1e6], device=self.device)

        cost = self.reg_cost(
            self.bboxes_centers,
            padded_gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )
        expected_cost = self.reg_cost(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )
        self.assertTrue(torch.equal(cost, expected_cost))

    def test_bbox_bev_l1_cost_normalizes_by_the_detector_range(self) -> None:
        """Test that the cost is the range-normalized L1 distance, not the metric one."""
        bboxes_centers = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, device=self.device)
        gt_bboxes_centers = torch.tensor([[[8.0, 0.0]]], dtype=torch.float32, device=self.device)
        valid_masks = torch.ones((1, 1), dtype=torch.bool, device=self.device)

        cost = BBoxBEVL1Cost(weight=1.0)(
            bboxes_centers, gt_bboxes_centers, valid_masks, self.point_cloud_range
        )

        # 8 m over an 80 m x extent is 0.1 in normalized coordinates.
        self.assertAlmostEqual(float(cost[0, 0, 0]), 0.1, places=6)
        # Halving the extent doubles the normalized distance.
        narrow_range = [-20.0, -40.0, -3.0, 20.0, 40.0, 5.0]
        narrow_cost = BBoxBEVL1Cost(weight=1.0)(
            bboxes_centers, gt_bboxes_centers, valid_masks, narrow_range
        )
        self.assertAlmostEqual(float(narrow_cost[0, 0, 0]), 0.2, places=6)

    def test_bbox_bev_l1_cost_scales_with_weight(self) -> None:
        """Test that the weight scales only the valid entries of the cost."""
        unit_cost = BBoxBEVL1Cost(weight=1.0)(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )
        scaled_cost = BBoxBEVL1Cost(weight=0.25)(
            self.bboxes_centers,
            self.gt_bboxes_centers,
            self.valid_masks,
            self.point_cloud_range,
        )

        valid_columns = self.valid_masks.unsqueeze(1).expand_as(unit_cost)
        self.assertTrue(
            torch.allclose(scaled_cost[valid_columns], unit_cost[valid_columns] * 0.25, atol=1e-6)
        )


class TestIoU3DCost(unittest.TestCase):
    """Unit tests for the IoU3DCost function."""

    def test_iou_3d_cost_negates_and_scales_the_iou(self) -> None:
        """Test that higher overlaps map to lower costs."""
        iou = torch.tensor([[0.0, 0.5], [0.9, 1.0]], dtype=torch.float32)

        cost = IoU3DCost(weight=0.25)(iou)

        self.assertEqual(cost.shape, iou.shape)
        self.assertTrue(torch.allclose(cost, -iou * 0.25, atol=1e-6))
        self.assertLess(float(cost[1, 1]), float(cost[0, 0]))


if __name__ == "__main__":
    unittest.main()
