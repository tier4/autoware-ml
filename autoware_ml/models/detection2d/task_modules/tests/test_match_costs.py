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

"""Unit tests for the matching costs of the 2D Hungarian assigner."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.models.detection2d.task_modules.match_costs import (
    BBoxL1Cost2D,
    Center2DL1Cost,
    IoUCost2D,
)


class TestMatchCosts2D(unittest.TestCase):
    """The cost terms by hand, so a wrong weight, sign or box form does not pass unnoticed."""

    def test_box_l1_cost_converts_the_ground_truth_to_the_proposal_form(self) -> None:
        """Test that the box L1 cost converts the side the box form names.

        Input: the proposal (0.5, 0.5, 0.2, 0.2), which is (0.4, 0.4) - (0.6, 0.6), against
        that box and (0.4, 0.4) - (0.8, 0.8), which is (0.6, 0.6, 0.4, 0.4); weight 5.
        Expected: in the default ``xywh`` form the ground truth is converted, so the L1 is 0
        and 0.6 and the costs 0 and 3. In the ``xyxy`` form the proposal is converted instead,
        and the L1 to the second box is 0.4, cost 2.
        Check: the cost of both forms.
        """
        bbox_pred = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        gt_bboxes = torch.tensor([[[0.4, 0.4, 0.6, 0.6], [0.4, 0.4, 0.8, 0.8]]])

        torch.testing.assert_close(
            BBoxL1Cost2D(weight=5.0)(bbox_pred, gt_bboxes), torch.tensor([[[0.0, 3.0]]])
        )
        torch.testing.assert_close(
            BBoxL1Cost2D(weight=5.0, box_format="xyxy")(bbox_pred, gt_bboxes[:, 1:]),
            torch.tensor([[[2.0]]]),
        )

    def test_iou_cost_is_the_negative_overlap(self) -> None:
        """Test that the overlap cost is the negative weighted GIoU or IoU.

        Input: (0, 0) - (2, 2) against (1, 1) - (3, 3), weight 2.
        Expected: the pair has GIoU 1/7 - 2/9 and IoU 1/7, so it costs minus twice those.
        Check: the cost of both modes.
        """
        boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
        gt_bboxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])

        torch.testing.assert_close(
            IoUCost2D(weight=2.0)(boxes, gt_bboxes), torch.tensor([[[-(1 / 7 - 2 / 9) * 2]]])
        )
        torch.testing.assert_close(
            IoUCost2D(weight=2.0, iou_mode="iou")(boxes, gt_bboxes), torch.tensor([[[-2 / 7]]])
        )

    def test_iou_cost_equals_the_pairwise_overlaps_of_every_image(self) -> None:
        """Test that the batched overlap cost is the pairwise overlap of every image on its own.

        Input: three images of 7 random proposals against 5 random boxes, a zero-area box
        among them, in both modes.
        Expected: the cost of an image is minus its ``BaseBBoxes2D`` pairwise overlaps, the
        images not mixing.
        Check: the cost of every image against the pairwise GIoU and IoU of its boxes.
        """
        generator = torch.Generator().manual_seed(1)
        corners = torch.rand(3, 7, 2, generator=generator) * 100
        boxes = torch.cat([corners, corners + torch.rand(3, 7, 2, generator=generator) * 50], -1)
        gt_corners = torch.rand(3, 5, 2, generator=generator) * 100
        gt_bboxes = torch.cat(
            [gt_corners, gt_corners + torch.rand(3, 5, 2, generator=generator) * 50], -1
        )
        gt_bboxes[1, 2, 2:] = gt_bboxes[1, 2, :2]

        for mode in ("giou", "iou"):
            cost = IoUCost2D(weight=1.0, iou_mode=mode)(boxes, gt_bboxes)
            for image in range(3):
                with self.subTest(mode=mode, image=image):
                    image_boxes, image_gt = (
                        BaseBBoxes2D(boxes[image]),
                        BaseBBoxes2D(gt_bboxes[image]),
                    )
                    overlaps = (
                        image_boxes.giou(image_gt) if mode == "giou" else image_boxes.iou(image_gt)
                    )
                    torch.testing.assert_close(cost[image], -overlaps)

    def test_the_costs_reject_unknown_formats_when_built(self) -> None:
        """Test that an unknown box form or overlap mode is refused when the cost is built.

        Input: a box L1 cost in the ``polygon`` form and an overlap cost in the ``iof`` mode.
        Expected: both are refused when they are built, before any matching.
        Check: the ValueError naming the field.
        """
        with self.assertRaisesRegex(ValueError, "box_format"):
            BBoxL1Cost2D(box_format="polygon")
        with self.assertRaisesRegex(ValueError, "iou_mode"):
            IoUCost2D(iou_mode="iof")

    def test_center_cost_is_the_weighted_l1_distance(self) -> None:
        """Test that the center cost is the weighted L1 distance.

        Input: the center (0.5, 0.5) against (0.6, 0.3), weight 10.
        Expected: the two are 0.1 + 0.2 = 0.3 apart in L1, so the cost is 3.
        Check: the cost.
        """
        torch.testing.assert_close(
            Center2DL1Cost(weight=10.0)(torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[0.6, 0.3]]])),
            torch.tensor([[[3.0]]]),
        )


if __name__ == "__main__":
    unittest.main()
