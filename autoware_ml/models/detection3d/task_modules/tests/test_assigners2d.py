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


"""Tests for the Hungarian assigner of the auxiliary 2D head."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.assigners2d import (
    AssignResult2D,
    BBoxL1Cost2D,
    HungarianAssigner2D,
    IoUCost2D,
)

IMAGE_HEIGHT, IMAGE_WIDTH = 100, 200


def _proposals_at(centers: list[tuple[float, float]], num_classes: int = 3):
    """Normalized ``(cx, cy, w, h)`` proposals at the given pixel centers, 20x20 px each."""
    boxes = torch.tensor(
        [
            [x / IMAGE_WIDTH, y / IMAGE_HEIGHT, 20 / IMAGE_WIDTH, 20 / IMAGE_HEIGHT]
            for x, y in centers
        ]
    )
    logits = torch.full((len(centers), num_classes), -4.0)
    return boxes, logits, boxes[:, :2].clone()


def _assign(bbox_pred, cls_pred, centers, gt_bboxes, gt_labels, gt_centers=None):
    """Run the default assigner; the centers default to the box centers."""
    if gt_centers is None:
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
    return HungarianAssigner2D().assign(
        bbox_pred, cls_pred, centers, gt_bboxes, gt_labels, gt_centers, IMAGE_HEIGHT, IMAGE_WIDTH
    )


class TestHungarianAssigner2D(unittest.TestCase):
    """Unit tests for HungarianAssigner2D."""

    def test_matches_every_ground_truth_box_to_the_proposal_on_top_of_it(self) -> None:
        """The third proposal, away from both boxes, is background: index 0 and label -1."""
        bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)])
        # Proposal 1 is confident about class 2, proposal 0 about class 0.
        cls_pred[1, 2] = 4.0
        cls_pred[0, 0] = 4.0
        gt_bboxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])

        result = _assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([2, 0]))

        self.assertIsInstance(result, AssignResult2D)
        self.assertEqual(result.num_gts, 2)
        self.assertEqual(result.gt_inds.tolist(), [2, 1, 0])
        self.assertEqual(result.labels.tolist(), [0, 2, -1])

    def test_an_image_without_ground_truth_assigns_every_proposal_to_background(self) -> None:
        bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0)])

        result = _assign(
            bbox_pred,
            cls_pred,
            centers,
            torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2),
        )

        self.assertEqual(result.num_gts, 0)
        self.assertEqual(result.gt_inds.tolist(), [0, 0])
        self.assertEqual(result.labels.tolist(), [-1, -1])

    def test_an_image_without_proposals_gives_empty_assignments(self) -> None:
        gt_bboxes = torch.tensor([[20.0, 20.0, 40.0, 40.0]])

        result = _assign(
            torch.zeros(0, 4), torch.zeros(0, 3), torch.zeros(0, 2), gt_bboxes, torch.tensor([1])
        )

        self.assertEqual(result.num_gts, 1)
        self.assertEqual(result.gt_inds.shape, (0,))
        self.assertEqual(result.labels.shape, (0,))

    def test_more_ground_truth_than_proposals_matches_every_proposal(self) -> None:
        """With three boxes and two proposals, both proposals get the box on top of them."""
        bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0)])
        gt_bboxes = torch.tensor(
            [[140.0, 50.0, 160.0, 70.0], [80.0, 40.0, 100.0, 60.0], [20.0, 20.0, 40.0, 40.0]]
        )

        result = _assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([0, 1, 2]))

        self.assertEqual(result.num_gts, 3)
        self.assertEqual(result.gt_inds.tolist(), [3, 1])
        self.assertEqual(result.labels.tolist(), [2, 0])

    def test_the_matching_is_globally_optimal(self) -> None:
        """Two proposals near box 1: the first a bit closer, the second much farther from box 2.

        Taking box 1 for the first proposal would leave the second proposal 110 px from box 2;
        the global optimum gives box 1 to the second proposal instead.
        """
        bbox_pred, cls_pred, centers = _proposals_at([(60.0, 50.0), (40.0, 50.0)])
        gt_bboxes = torch.tensor([[40.0, 40.0, 60.0, 60.0], [140.0, 40.0, 160.0, 60.0]])

        result = _assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([0, 0]))

        self.assertEqual(result.gt_inds.tolist(), [2, 1])

    def test_reduced_precision_inputs_are_matched_like_their_float32_values(self) -> None:
        """Two proposals on top of two boxes of classes 0 and 1, so the classes decide the match.

        Autocast gives bfloat16 logits, whose sigmoid rounds to 1 for both proposals, and
        bfloat16 boxes, which are off by up to 1/256 of the image.
        """
        bbox_pred, _, centers = _proposals_at([(100.0, 50.0), (100.0, 50.0)], num_classes=2)
        # Proposal 0 is the more confident about class 1, proposal 1 about class 0.
        logits = torch.tensor([[7.0, 8.0], [8.0, 7.0]])
        gt_bboxes = torch.tensor([[90.0, 40.0, 110.0, 60.0], [90.0, 40.0, 110.0, 60.0]])
        gt_labels = torch.tensor([0, 1])

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                result = _assign(
                    bbox_pred.to(dtype),
                    logits.to(dtype),
                    centers.to(dtype),
                    gt_bboxes.to(dtype),
                    gt_labels,
                )

                self.assertEqual(result.gt_inds.tolist(), [2, 1])

    def test_a_negative_label_is_rejected(self) -> None:
        """-1 is the padding of a batched ground truth; the 2D ground truth is not padded."""
        bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0)])
        gt_bboxes = torch.tensor([[20.0, 20.0, 40.0, 40.0]])

        with self.assertRaisesRegex(ValueError, "negative"):
            _assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([-1]))

    def test_a_label_beyond_the_class_channels_is_rejected(self) -> None:
        bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0)], num_classes=3)
        gt_bboxes = torch.tensor([[20.0, 20.0, 40.0, 40.0]])

        with self.assertRaises(ValueError):
            _assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([3]))

    def test_the_costs_reject_unknown_formats_when_built(self) -> None:
        with self.assertRaisesRegex(ValueError, "box_format"):
            BBoxL1Cost2D(box_format="xywhr")
        with self.assertRaisesRegex(ValueError, "iou_mode"):
            IoUCost2D(iou_mode="iof")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_cuda_inputs_give_the_cpu_assignment_on_the_input_device(self) -> None:
        bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)])
        cls_pred[1, 2] = 4.0
        cls_pred[0, 0] = 4.0
        gt_bboxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])
        gt_labels = torch.tensor([2, 0])

        expected = _assign(bbox_pred, cls_pred, centers, gt_bboxes, gt_labels)
        result = _assign(
            bbox_pred.cuda(), cls_pred.cuda(), centers.cuda(), gt_bboxes.cuda(), gt_labels.cuda()
        )

        self.assertEqual(result.gt_inds.device.type, "cuda")
        self.assertEqual(result.gt_inds.tolist(), expected.gt_inds.tolist())
        self.assertEqual(result.labels.tolist(), expected.labels.tolist())


if __name__ == "__main__":
    unittest.main()
