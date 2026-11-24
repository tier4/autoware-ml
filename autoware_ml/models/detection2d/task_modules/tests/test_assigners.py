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

"""Unit tests for the 2D Hungarian assigner."""

from __future__ import annotations

import unittest

from jaxtyping import Bool, Float32, Int64
import torch

from autoware_ml.models.detection2d.task_modules.assigners import (
    AssignResult2D,
    HungarianAssigner2D,
)


IMAGE_HEIGHT, IMAGE_WIDTH = 100, 200


class TestHungarianAssigner2D(unittest.TestCase):
    """Unit tests for HungarianAssigner2D on 200 x 100 px images."""

    def _proposals_at(
        self, centers: list[tuple[float, float]], num_classes: int = 3
    ) -> tuple[
        Float32[torch.Tensor, "num_queries 4"],
        Float32[torch.Tensor, "num_queries num_classes"],
        Float32[torch.Tensor, "num_queries 2"],
    ]:
        """Build 20 x 20 px proposals at the given pixel centers, with logits of -4.

        Args:
            centers: Pixel center of every proposal.
            num_classes: Number of class logits of every proposal.

        Returns:
            The normalized ``(cx, cy, w, h)`` boxes, the logits and the normalized centers.
        """
        boxes = torch.tensor(
            [
                [x / IMAGE_WIDTH, y / IMAGE_HEIGHT, 20 / IMAGE_WIDTH, 20 / IMAGE_HEIGHT]
                for x, y in centers
            ]
        )
        logits = torch.full((len(centers), num_classes), -4.0)
        return boxes, logits, boxes[:, :2].clone()

    def _assign(
        self,
        bbox_pred: Float32[torch.Tensor, "num_queries 4"],
        cls_pred: Float32[torch.Tensor, "num_queries num_classes"],
        centers: Float32[torch.Tensor, "num_queries 2"],
        gt_bboxes: Float32[torch.Tensor, "num_gts 4"],
        gt_labels: Int64[torch.Tensor, " num_gts"],
        gt_centers: Float32[torch.Tensor, "num_gts 2"] | None = None,
    ) -> AssignResult2D:
        """Run the default assigner on one image.

        Args:
            bbox_pred: Normalized ``(cx, cy, w, h)`` proposals.
            cls_pred: Proposal logits.
            centers: Normalized proposal centers.
            gt_bboxes: ``(x1, y1, x2, y2)`` ground-truth boxes in pixels.
            gt_labels: Ground-truth labels.
            gt_centers: Ground-truth centers in pixels; the box centers when None.

        Returns:
            The assignment of the image, as a batch of one.
        """
        if gt_centers is None:
            gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        return HungarianAssigner2D().assign(
            bbox_pred.unsqueeze(0),
            cls_pred.unsqueeze(0),
            centers.unsqueeze(0),
            gt_bboxes.unsqueeze(0),
            gt_labels.unsqueeze(0),
            gt_centers.unsqueeze(0),
            torch.ones(1, gt_bboxes.shape[0], dtype=torch.bool, device=gt_bboxes.device),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )

    def _pad(
        self,
        images: list[tuple[Float32[torch.Tensor, "num_gts 4"], Int64[torch.Tensor, " num_gts"]]],
    ) -> tuple[
        Float32[torch.Tensor, "num_images max_num_gts 4"],
        Int64[torch.Tensor, "num_images max_num_gts"],
        Float32[torch.Tensor, "num_images max_num_gts 2"],
        Bool[torch.Tensor, "num_images max_num_gts"],
    ]:
        """Pad the boxes and labels of several images to the largest count, -1 on the padding.

        Args:
            images: The ``(x1, y1, x2, y2)`` pixel boxes and the labels of every image.

        Returns:
            The padded boxes, labels and box centers, and the mask of the real boxes.
        """
        max_num_gts = max(boxes.shape[0] for boxes, _ in images)
        gt_bboxes = torch.zeros(len(images), max_num_gts, 4)
        gt_labels = torch.full((len(images), max_num_gts), -1, dtype=torch.long)
        valid_masks = torch.zeros(len(images), max_num_gts, dtype=torch.bool)
        for image, (boxes, labels) in enumerate(images):
            gt_bboxes[image, : boxes.shape[0]] = boxes
            gt_labels[image, : boxes.shape[0]] = labels
            valid_masks[image, : boxes.shape[0]] = True
        return gt_bboxes, gt_labels, (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2, valid_masks

    def test_the_default_assigner_carries_the_reference_weights(self) -> None:
        """Test that the default assigner carries the Focal-PETR matching weights.

        Input: the assigner built without arguments.
        Expected: the Focal-PETR matching weights: classification 2, box L1 5 on
        (cx, cy, w, h), GIoU 2 and center L1 10.
        Check: the weight, form and mode of every cost.
        """
        assigner = HungarianAssigner2D()

        self.assertEqual(assigner.cls_cost.weight, 2.0)
        self.assertEqual((assigner.reg_cost.weight, assigner.reg_cost.box_format), (5.0, "xywh"))
        self.assertEqual((assigner.iou_cost.weight, assigner.iou_cost.iou_mode), (2.0, "giou"))
        self.assertEqual(assigner.centers2d_cost.weight, 10.0)

    def test_matches_every_ground_truth_box_to_the_proposal_on_top_of_it(self) -> None:
        """Test that every box goes to the proposal on top of it.

        Input: three proposals, two on top of a class 2 and a class 0 box and confident about
        those classes, the third away from both boxes.
        Expected: each box goes to the proposal on top of it; the third proposal is background.
        Check: the result type, the box count, the 1-based box indices (0 for background) and
        the labels (-1 for background).
        """
        bbox_pred, cls_pred, centers = self._proposals_at(
            [(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)]
        )
        # Proposal 1 is confident about class 2, proposal 0 about class 0.
        cls_pred[1, 2] = 4.0
        cls_pred[0, 0] = 4.0
        gt_bboxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])

        result = self._assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([2, 0]))

        self.assertIsInstance(result, AssignResult2D)
        self.assertEqual(result.num_gts.tolist(), [2])
        self.assertEqual(result.gt_inds.tolist(), [[2, 1, 0]])
        self.assertEqual(result.labels.tolist(), [[0, 2, -1]])

    def test_images_are_matched_on_their_own_boxes(self) -> None:
        """Test that a batch of images gives every image the assignment of its own boxes.

        Input: the three proposals of the matching test on three images: one with its two
        boxes, one without a box, and one with the two boxes and a third one no proposal is
        near, so the ground truth is padded to three columns.
        Expected: the assignment of every image equals the one of that image alone, so the
        padding columns are never matched and an image without a box is all background.
        Check: the box counts, and the box indices and labels of every image against the
        single-image assignment.
        """
        bbox_pred, cls_pred, centers = self._proposals_at(
            [(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)]
        )
        cls_pred[1, 2] = 4.0
        cls_pred[0, 0] = 4.0
        two_boxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])
        three_boxes = torch.cat([two_boxes, torch.tensor([[180.0, 80.0, 200.0, 100.0]])])
        images = [
            (two_boxes, torch.tensor([2, 0])),
            (two_boxes[:0], torch.tensor([], dtype=torch.long)),
            (three_boxes, torch.tensor([2, 0, 1])),
        ]
        gt_bboxes, gt_labels, gt_centers, valid_masks = self._pad(images)

        result = HungarianAssigner2D().assign(
            bbox_pred.expand(3, -1, -1),
            cls_pred.expand(3, -1, -1),
            centers.expand(3, -1, -1),
            gt_bboxes,
            gt_labels,
            gt_centers,
            valid_masks,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )

        self.assertEqual(result.num_gts.tolist(), [2, 0, 3])
        for image, (boxes, labels) in enumerate(images):
            with self.subTest(image=image):
                alone = self._assign(bbox_pred, cls_pred, centers, boxes, labels)
                self.assertEqual(result.gt_inds[image].tolist(), alone.gt_inds[0].tolist())
                self.assertEqual(result.labels[image].tolist(), alone.labels[0].tolist())
        self.assertEqual(result.gt_inds[1].tolist(), [0, 0, 0])

    def test_an_image_without_ground_truth_assigns_every_proposal_to_background(self) -> None:
        """Test that an image without a box gives every proposal to the background.

        Input: two proposals and no ground truth.
        Expected: both proposals are background.
        Check: no boxes, box index 0 and label -1 for both proposals.
        """
        bbox_pred, cls_pred, centers = self._proposals_at([(30.0, 30.0), (150.0, 60.0)])

        result = self._assign(
            bbox_pred,
            cls_pred,
            centers,
            torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2),
        )

        self.assertEqual(result.num_gts.tolist(), [0])
        self.assertEqual(result.gt_inds.tolist(), [[0, 0]])
        self.assertEqual(result.labels.tolist(), [[-1, -1]])

    def test_an_image_without_proposals_gives_empty_assignments(self) -> None:
        """Test that an image without a proposal gives nothing to assign.

        Input: one ground-truth box and no proposals.
        Expected: nothing to assign.
        Check: one box, and empty box indices and labels.
        """
        gt_bboxes = torch.tensor([[20.0, 20.0, 40.0, 40.0]])

        result = self._assign(
            torch.zeros(0, 4), torch.zeros(0, 3), torch.zeros(0, 2), gt_bboxes, torch.tensor([1])
        )

        self.assertEqual(result.num_gts.tolist(), [1])
        self.assertEqual(result.gt_inds.shape, (1, 0))
        self.assertEqual(result.labels.shape, (1, 0))

    def test_more_ground_truth_than_proposals_matches_every_proposal(self) -> None:
        """Test that more boxes than proposals match every proposal and leave a box out.

        Input: three ground-truth boxes and two proposals, each on top of one of the boxes.
        Expected: both proposals get the box on top of them; the third box stays unmatched.
        Check: the box count, the box indices and the labels.
        """
        bbox_pred, cls_pred, centers = self._proposals_at([(30.0, 30.0), (150.0, 60.0)])
        gt_bboxes = torch.tensor(
            [[140.0, 50.0, 160.0, 70.0], [80.0, 40.0, 100.0, 60.0], [20.0, 20.0, 40.0, 40.0]]
        )

        result = self._assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([0, 1, 2]))

        self.assertEqual(result.num_gts.tolist(), [3])
        self.assertEqual(result.gt_inds.tolist(), [[3, 1]])
        self.assertEqual(result.labels.tolist(), [[2, 0]])

    def test_the_matching_is_globally_optimal(self) -> None:
        """Test that the matching minimizes the total cost, not the cost of every proposal.

        Input: two proposals near box 1, the first a bit closer to it and the second much
        farther from box 2.
        Expected: taking box 1 for the first proposal would leave the second proposal 110 px
        from box 2, so the global optimum gives box 1 to the second proposal instead.
        Check: the box indices.
        """
        bbox_pred, cls_pred, centers = self._proposals_at([(60.0, 50.0), (40.0, 50.0)])
        gt_bboxes = torch.tensor([[40.0, 40.0, 60.0, 60.0], [140.0, 40.0, 160.0, 60.0]])

        result = self._assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([0, 0]))

        self.assertEqual(result.gt_inds.tolist(), [[2, 1]])

    def test_inputs_other_than_float32_are_rejected(self) -> None:
        """Test that the assigner matches float32 inputs only.

        Input: one proposal on top of one box, with each box, logit and center tensor in
        bfloat16 and float16 in turn, the others in float32.
        Expected: autocast gives bfloat16 logits, whose sigmoid rounds to 1 above logit 6.2,
        and bfloat16 boxes, off by up to 1/256 of the image, so the assigner matches only
        float32 inputs and leaves the cast to the caller.
        Check: the ValueError naming the tensor.
        """
        bbox_pred, cls_pred, centers = self._proposals_at([(30.0, 30.0)])
        gt_bboxes = torch.tensor([[[20.0, 20.0, 40.0, 40.0]]])
        inputs = {
            "bbox_pred": bbox_pred.unsqueeze(0),
            "cls_pred": cls_pred.unsqueeze(0),
            "pred_centers2d": centers.unsqueeze(0),
            "gt_bboxes": gt_bboxes,
            "gt_centers2d": (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2,
        }

        for name in inputs:
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(name=name, dtype=dtype):
                    arguments = dict(inputs)
                    arguments[name] = inputs[name].to(dtype)
                    with self.assertRaisesRegex(ValueError, name):
                        HungarianAssigner2D().assign(
                            arguments["bbox_pred"],
                            arguments["cls_pred"],
                            arguments["pred_centers2d"],
                            arguments["gt_bboxes"],
                            torch.tensor([[0]]),
                            arguments["gt_centers2d"],
                            torch.ones(1, 1, dtype=torch.bool),
                            IMAGE_HEIGHT,
                            IMAGE_WIDTH,
                        )

    def test_a_negative_label_on_a_real_box_is_rejected(self) -> None:
        """Test that -1 marks the padding only.

        Input: one box labelled -1 and marked valid, then the same box marked padding.
        Expected: -1 is the label of the padding, so a valid box carrying it is refused instead
        of being matched, while the padding is matched to nothing.
        Check: the ValueError naming the negative label, and a background assignment.
        """
        bbox_pred, cls_pred, centers = self._proposals_at([(30.0, 30.0)])
        gt_bboxes = torch.tensor([[[20.0, 20.0, 40.0, 40.0]]])
        gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2

        with self.assertRaisesRegex(ValueError, "negative"):
            self._assign(bbox_pred, cls_pred, centers, gt_bboxes[0], torch.tensor([-1]))
        result = HungarianAssigner2D().assign(
            bbox_pred.unsqueeze(0),
            cls_pred.unsqueeze(0),
            centers.unsqueeze(0),
            gt_bboxes,
            torch.tensor([[-1]]),
            gt_centers,
            torch.zeros(1, 1, dtype=torch.bool),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        self.assertEqual(result.num_gts.tolist(), [0])
        self.assertEqual(result.gt_inds.tolist(), [[0]])

    def test_a_label_beyond_the_class_channels_is_rejected(self) -> None:
        """Test that a label without a class channel is refused.

        Input: one box labelled 3 against logits of three classes.
        Expected: the label has no class channel to read.
        Check: the ValueError.
        """
        bbox_pred, cls_pred, centers = self._proposals_at([(30.0, 30.0)], num_classes=3)
        gt_bboxes = torch.tensor([[20.0, 20.0, 40.0, 40.0]])

        with self.assertRaises(ValueError):
            self._assign(bbox_pred, cls_pred, centers, gt_bboxes, torch.tensor([3]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_cuda_inputs_give_the_cpu_assignment_on_the_input_device(self) -> None:
        """Test that CUDA inputs are matched on the host and the result returned on CUDA.

        Input: the three proposals and two boxes of the matching test, on the CPU and on CUDA.
        Expected: the CUDA inputs are matched on the CPU and the result moved back to CUDA.
        Check: the device of the result and the box indices and labels of both runs.
        """
        bbox_pred, cls_pred, centers = self._proposals_at(
            [(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)]
        )
        cls_pred[1, 2] = 4.0
        cls_pred[0, 0] = 4.0
        gt_bboxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])
        gt_labels = torch.tensor([2, 0])

        expected = self._assign(bbox_pred, cls_pred, centers, gt_bboxes, gt_labels)
        result = self._assign(
            bbox_pred.cuda(), cls_pred.cuda(), centers.cuda(), gt_bboxes.cuda(), gt_labels.cuda()
        )

        self.assertEqual(result.gt_inds.device.type, "cuda")
        self.assertEqual(result.gt_inds.tolist(), expected.gt_inds.tolist())
        self.assertEqual(result.labels.tolist(), expected.labels.tolist())


if __name__ == "__main__":
    unittest.main()
