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

"""Unit tests for the detection3d proposal assigners."""

import unittest
from typing import Sequence

from jaxtyping import Bool, Float32
import torch

from autoware_ml.models.detection3d.task_modules.assigners import (
    AssignResult,
    HungarianAssigner3D,
    _bev_iou_aligned,
)
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.types.geometry import Box3DFieldIndex


class TestBEVIoUAligned(unittest.TestCase):
    """Unit tests for the _bev_iou_aligned function."""

    def setUp(self) -> None:
        """Set up the same proposal boxes, gt boxes and masks for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 2
        self.num_bboxes = 4
        self.num_gt_bboxes = 3

        # (batch_size, num_bboxes, code_size) as (center_x, center_y, length, width)
        self.boxes = self._make_boxes(
            [
                [
                    [0.0, 0.0, 4.0, 2.0],  # exactly on gt 0 of sample 0
                    [2.0, 0.0, 4.0, 2.0],  # half a length away from gt 0
                    [50.0, 50.0, 4.0, 2.0],  # disjoint from every gt
                    [10.0, 10.0, 8.0, 4.0],  # encloses gt 1 of sample 0
                ],
                [
                    [-5.0, 3.0, 6.0, 3.0],
                    [0.0, 0.0, 2.0, 2.0],
                    [20.0, -20.0, 4.0, 4.0],
                    [-5.0, 3.0, 3.0, 1.5],
                ],
            ]
        )
        # (batch_size, num_gt_bboxes, code_size)
        self.gt_boxes = self._make_boxes(
            [
                [
                    [0.0, 0.0, 4.0, 2.0],
                    [10.0, 10.0, 4.0, 2.0],
                    [-30.0, 0.0, 4.0, 2.0],
                ],
                [
                    [-5.0, 3.0, 6.0, 3.0],
                    [20.0, -20.0, 4.0, 4.0],
                    [1.0, 0.0, 2.0, 2.0],
                ],
            ]
        )
        # (batch_size, num_gt_bboxes)
        self.valid_masks = torch.tensor(
            [[True, True, False], [True, True, True]], device=self.device
        )

    def _make_boxes(
        self, boxes: Sequence[Sequence[Sequence[float]]]
    ) -> Float32[torch.Tensor, "batch_size num_bboxes code_size"]:
        """Build (center_x, center_y, length, width) specs into full code_size box tensors.

        The remaining fields (center_z, height, yaw, velocity) are filled with values the
        axis-aligned BEV IoU must ignore, so a regression that reads the wrong field axis shows
        up as a wrong IoU rather than passing silently.
        """
        batch_size = len(boxes)
        num_bboxes = len(boxes[0])
        code_size = int(Box3DFieldIndex.VELOCITY_Y) + 1
        box_tensor = torch.zeros((batch_size, num_bboxes, code_size), dtype=torch.float32)
        for batch_index, sample_boxes in enumerate(boxes):
            for box_index, (center_x, center_y, length, width) in enumerate(sample_boxes):
                box_tensor[batch_index, box_index, Box3DFieldIndex.X] = center_x
                box_tensor[batch_index, box_index, Box3DFieldIndex.Y] = center_y
                box_tensor[batch_index, box_index, Box3DFieldIndex.LENGTH] = length
                box_tensor[batch_index, box_index, Box3DFieldIndex.WIDTH] = width
                # Distinctive values in the fields the BEV IoU must not read.
                box_tensor[batch_index, box_index, Box3DFieldIndex.Z] = 100.0 + box_index
                box_tensor[batch_index, box_index, Box3DFieldIndex.HEIGHT] = 7.0 + box_index
                box_tensor[batch_index, box_index, Box3DFieldIndex.YAW] = 1.1
                box_tensor[batch_index, box_index, Box3DFieldIndex.VELOCITY_X] = -3.0
                box_tensor[batch_index, box_index, Box3DFieldIndex.VELOCITY_Y] = 4.0
        return box_tensor.to(self.device)

    def _expected_bev_iou(
        self,
        boxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        gt_boxes: Float32[torch.Tensor, "batch_size max_num_gt_bboxes code_size"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes max_num_gt_bboxes"]:
        """Build the reference IoU with an explicit per-sample, per-pair Python loop."""
        batch_size, num_bboxes, _ = boxes.shape
        num_gt_bboxes = gt_boxes.shape[1]
        ious = torch.zeros(
            (batch_size, num_bboxes, num_gt_bboxes), dtype=torch.float32, device=boxes.device
        )
        for batch_index in range(batch_size):
            for box_index in range(num_bboxes):
                for gt_index in range(num_gt_bboxes):
                    if not bool(valid_masks[batch_index, gt_index]):
                        continue
                    box = boxes[batch_index, box_index]
                    gt_box = gt_boxes[batch_index, gt_index]
                    box_length = max(float(box[Box3DFieldIndex.LENGTH]), 0.0)
                    box_width = max(float(box[Box3DFieldIndex.WIDTH]), 0.0)
                    gt_length = max(float(gt_box[Box3DFieldIndex.LENGTH]), 0.0)
                    gt_width = max(float(gt_box[Box3DFieldIndex.WIDTH]), 0.0)

                    box_x_min = float(box[Box3DFieldIndex.X]) - box_length * 0.5
                    box_x_max = float(box[Box3DFieldIndex.X]) + box_length * 0.5
                    box_y_min = float(box[Box3DFieldIndex.Y]) - box_width * 0.5
                    box_y_max = float(box[Box3DFieldIndex.Y]) + box_width * 0.5
                    gt_x_min = float(gt_box[Box3DFieldIndex.X]) - gt_length * 0.5
                    gt_x_max = float(gt_box[Box3DFieldIndex.X]) + gt_length * 0.5
                    gt_y_min = float(gt_box[Box3DFieldIndex.Y]) - gt_width * 0.5
                    gt_y_max = float(gt_box[Box3DFieldIndex.Y]) + gt_width * 0.5

                    inter_width = max(min(box_x_max, gt_x_max) - max(box_x_min, gt_x_min), 0.0)
                    inter_height = max(min(box_y_max, gt_y_max) - max(box_y_min, gt_y_min), 0.0)
                    inter_area = inter_width * inter_height
                    union = box_length * box_width + gt_length * gt_width - inter_area
                    ious[batch_index, box_index, gt_index] = inter_area / max(union, 1e-6)
        return ious

    def test_bev_iou_aligned_matches_per_sample_reference(self) -> None:
        """Test that the batched IoU matches an explicit per-pair reference loop."""
        ious = _bev_iou_aligned(self.boxes, self.gt_boxes, self.valid_masks)

        self.assertEqual(ious.shape, (self.batch_size, self.num_bboxes, self.num_gt_bboxes))
        expected_ious = self._expected_bev_iou(self.boxes, self.gt_boxes, self.valid_masks)
        self.assertTrue(torch.allclose(ious, expected_ious, atol=1e-6))

    def test_bev_iou_aligned_is_independent_across_batch(self) -> None:
        """Test that each sample's IoU block only depends on that sample's inputs."""
        batched_ious = _bev_iou_aligned(self.boxes, self.gt_boxes, self.valid_masks)

        for batch_index in range(self.batch_size):
            single_ious = _bev_iou_aligned(
                self.boxes[batch_index : batch_index + 1],
                self.gt_boxes[batch_index : batch_index + 1],
                self.valid_masks[batch_index : batch_index + 1],
            )
            self.assertTrue(
                torch.allclose(single_ious[0], batched_ious[batch_index], atol=1e-6),
                msg=f"batch element {batch_index} depends on the rest of the batch",
            )

    def test_bev_iou_aligned_known_overlaps(self) -> None:
        """Test identical, partially overlapping, enclosed and disjoint box pairs."""
        ious = _bev_iou_aligned(self.boxes, self.gt_boxes, self.valid_masks)

        # Proposal 0 is exactly gt 0.
        self.assertAlmostEqual(float(ious[0, 0, 0]), 1.0, places=6)
        # Proposal 1 is shifted by half a length, so the 4x2 boxes share a 2x2 area over a union
        # of 8 + 8 - 4 = 12 cells: 4 / 12 = 1/3.
        self.assertAlmostEqual(float(ious[0, 1, 0]), 1.0 / 3.0, places=6)
        # Proposal 2 is far away from every gt.
        self.assertTrue(torch.all(ious[0, 2, :] == 0.0))
        # Proposal 3 is an 8x4 box centered on the 4x2 gt 1, so the gt is fully enclosed and the
        # IoU is the area ratio 8 / 32 = 0.25.
        self.assertAlmostEqual(float(ious[0, 3, 1]), 0.25, places=6)
        # Sample 1 proposal 3 is gt 0 scaled by half about the same center: 4.5 / 18 = 0.25.
        self.assertAlmostEqual(float(ious[1, 3, 0]), 0.25, places=6)

    def test_bev_iou_aligned_masks_invalid_gt_bboxes(self) -> None:
        """Test that padded gt columns are zeroed even when they overlap a proposal."""
        # Make the padded gt of sample 0 coincide with proposal 0, so only the mask can zero it.
        gt_boxes = self.gt_boxes.clone()
        gt_boxes[0, 2] = self.boxes[0, 0]

        ious = _bev_iou_aligned(self.boxes, gt_boxes, self.valid_masks)

        invalid_columns = ~self.valid_masks.unsqueeze(1).expand_as(ious)
        self.assertTrue(torch.all(ious[invalid_columns] == 0.0))
        # The valid columns are untouched by the padded gt's geometry.
        expected_ious = _bev_iou_aligned(self.boxes, self.gt_boxes, self.valid_masks)
        self.assertTrue(torch.allclose(ious, expected_ious, atol=1e-6))

    def test_bev_iou_aligned_ignores_non_bev_fields(self) -> None:
        """Test that z, height, yaw and velocity do not change the axis-aligned BEV IoU."""
        boxes = self.boxes.clone()
        gt_boxes = self.gt_boxes.clone()
        for field in (
            Box3DFieldIndex.Z,
            Box3DFieldIndex.HEIGHT,
            Box3DFieldIndex.YAW,
            Box3DFieldIndex.VELOCITY_X,
            Box3DFieldIndex.VELOCITY_Y,
        ):
            boxes[..., field] = -42.0
            gt_boxes[..., field] = 17.0

        ious = _bev_iou_aligned(boxes, gt_boxes, self.valid_masks)
        expected_ious = _bev_iou_aligned(self.boxes, self.gt_boxes, self.valid_masks)
        self.assertTrue(torch.allclose(ious, expected_ious, atol=1e-6))

    def test_bev_iou_aligned_clamps_degenerate_extents(self) -> None:
        """Test that zero and negative box extents yield a finite zero IoU instead of NaN."""
        boxes = self._make_boxes(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -4.0, -2.0], [0.0, 0.0, 4.0, 2.0]]]
        )
        gt_boxes = self._make_boxes([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 2.0]]])
        valid_masks = torch.ones((1, 2), dtype=torch.bool, device=self.device)

        ious = _bev_iou_aligned(boxes, gt_boxes, valid_masks)

        self.assertTrue(torch.all(ious.isfinite()))
        # A degenerate proposal or gt has no area, so every pair involving one has zero IoU.
        self.assertTrue(torch.all(ious[0, 0, :] == 0.0))
        self.assertTrue(torch.all(ious[0, 1, :] == 0.0))
        self.assertTrue(torch.all(ious[0, :, 0] == 0.0))
        # The one well-formed pair still matches exactly.
        self.assertAlmostEqual(float(ious[0, 2, 1]), 1.0, places=6)

    def test_bev_iou_aligned_is_bounded(self) -> None:
        """Test that random boxes always produce IoUs inside [0, 1]."""
        torch.manual_seed(0)
        code_size = int(Box3DFieldIndex.VELOCITY_Y) + 1
        boxes = torch.randn(3, 6, code_size, device=self.device) * 10.0
        gt_boxes = torch.randn(3, 5, code_size, device=self.device) * 10.0
        valid_masks = torch.rand(3, 5, device=self.device) > 0.3

        ious = _bev_iou_aligned(boxes, gt_boxes, valid_masks)

        self.assertEqual(ious.shape, (3, 6, 5))
        self.assertTrue(torch.all(ious >= 0.0))
        self.assertTrue(torch.all(ious <= 1.0))
        self.assertTrue(torch.all(ious.isfinite()))


class TestHungarianAssigner3D(unittest.TestCase):
    """Unit tests for the HungarianAssigner3D class.

    Moved here from autoware_ml/tests/models/test_detection3d_task_modules.py and extended for
    the batched assign signature (batched tensors plus valid_masks, with point_cloud_range now a
    field on the assigner rather than an argument).
    """

    def setUp(self) -> None:
        """Set up an assigner and a two-sample batch with unambiguous best proposals."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 2
        self.num_bboxes = 3
        self.max_num_gt_bboxes = 2
        self.num_classes = 2
        self.point_cloud_range = [0.0, 0.0, -1.0, 40.0, 40.0, 3.0]
        self.assigner = HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=0.15),
            reg_cost=BBoxBEVL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.25),
            point_cloud_range=self.point_cloud_range,
        )

        # Sample 0 has two real gts; sample 1 has one real gt and one padded column.
        # (batch_size, num_bboxes, code_size)
        self.bboxes = self._boxes_from_centers(
            [
                [(2.0, 2.0), (20.0, 20.0), (35.0, 35.0)],
                [(5.0, 5.0), (30.0, 10.0), (1.0, 1.0)],
            ]
        )
        # (batch_size, max_num_gt_bboxes, code_size). Each gt sits essentially on one proposal:
        # sample 0 gt 0 on proposal 0 and gt 1 on proposal 1, sample 1 gt 0 on proposal 0.
        self.gt_bboxes = self._boxes_from_centers(
            [
                [(2.1, 2.0), (20.2, 20.0)],
                [(5.1, 5.0), (-99.0, -99.0)],
            ]
        )
        # (batch_size, max_num_gt_bboxes). The padded label is -1, as Detection3DGTBatch emits.
        self.gt_labels = torch.tensor([[1, 0], [0, -1]], dtype=torch.int64, device=self.device)
        self.valid_masks = torch.tensor([[True, True], [True, False]], device=self.device)
        # (batch_size, num_bboxes, num_classes). Each proposal is confident about the class of the
        # gt it sits on, so classification agrees with geometry instead of fighting it.
        self.cls_pred = torch.tensor(
            [
                [[-3.0, 4.0], [4.0, -3.0], [0.0, 0.0]],
                [[4.0, -3.0], [0.0, 0.0], [-1.0, -1.0]],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _boxes_from_centers(
        self, centers: Sequence[Sequence[tuple[float, float]]]
    ) -> Float32[torch.Tensor, "batch_size num_bboxes code_size"]:
        """Build BEV centers into full code_size boxes with a fixed 4 x 2 footprint."""
        batch_size = len(centers)
        num_bboxes = len(centers[0])
        code_size = int(Box3DFieldIndex.VELOCITY_Y) + 1
        box_tensor = torch.zeros((batch_size, num_bboxes, code_size), dtype=torch.float32)
        for batch_index, sample_centers in enumerate(centers):
            for box_index, (center_x, center_y) in enumerate(sample_centers):
                box_tensor[batch_index, box_index, Box3DFieldIndex.X] = center_x
                box_tensor[batch_index, box_index, Box3DFieldIndex.Y] = center_y
                box_tensor[batch_index, box_index, Box3DFieldIndex.Z] = 0.5
                box_tensor[batch_index, box_index, Box3DFieldIndex.LENGTH] = 4.0
                box_tensor[batch_index, box_index, Box3DFieldIndex.WIDTH] = 2.0
                box_tensor[batch_index, box_index, Box3DFieldIndex.HEIGHT] = 1.5
        return box_tensor.to(self.device)

    def _assign(self) -> AssignResult:
        """Run the assigner over the shared two-sample batch."""
        return self.assigner.assign(
            bboxes=self.bboxes,
            gt_bboxes=self.gt_bboxes,
            gt_labels=self.gt_labels,
            cls_pred=self.cls_pred,
            valid_masks=self.valid_masks,
        )

    def test_assign_matches_best_query(self) -> None:
        """Test that the proposal sitting on each gt wins that gt."""
        result = self._assign()

        # gt_inds is one-based, with 0 meaning "negative" and -1 meaning "ignore".
        self.assertEqual(result.gt_inds[0].tolist(), [1, 2, 0])
        self.assertEqual(result.labels[0].tolist(), [1, 0, -1])
        # Sample 1 has a single real gt, so only proposal 0 is positive.
        self.assertEqual(result.gt_inds[1].tolist(), [1, 0, 0])
        self.assertEqual(result.labels[1].tolist(), [0, -1, -1])

    def test_assign_returns_batched_shapes_and_dtypes(self) -> None:
        """Test the shape and dtype of every field of the assignment result."""
        result = self._assign()

        self.assertEqual(result.gt_inds.shape, (self.batch_size, self.num_bboxes))
        self.assertEqual(result.labels.shape, (self.batch_size, self.num_bboxes))
        assert result.max_overlaps is not None
        self.assertEqual(result.max_overlaps.shape, (self.batch_size, self.num_bboxes))
        self.assertEqual(result.gt_inds.dtype, torch.int64)
        self.assertEqual(result.labels.dtype, torch.int64)
        self.assertEqual(result.max_overlaps.dtype, torch.float32)
        # num_gts counts the real gts per sample, so the padded column of sample 1 is excluded.
        self.assertEqual(result.num_gts.shape, (self.batch_size,))
        self.assertEqual(result.num_gts.tolist(), [2, 1])

    def test_assign_is_independent_across_batch(self) -> None:
        """Test that each sample's assignment only depends on that sample's inputs."""
        batched_result = self._assign()

        for batch_index in range(self.batch_size):
            single_result = self.assigner.assign(
                bboxes=self.bboxes[batch_index : batch_index + 1],
                gt_bboxes=self.gt_bboxes[batch_index : batch_index + 1],
                gt_labels=self.gt_labels[batch_index : batch_index + 1],
                cls_pred=self.cls_pred[batch_index : batch_index + 1],
                valid_masks=self.valid_masks[batch_index : batch_index + 1],
            )
            self.assertEqual(
                single_result.gt_inds[0].tolist(),
                batched_result.gt_inds[batch_index].tolist(),
                msg=f"batch element {batch_index} depends on the rest of the batch",
            )
            self.assertEqual(
                single_result.labels[0].tolist(), batched_result.labels[batch_index].tolist()
            )
            assert single_result.max_overlaps is not None
            assert batched_result.max_overlaps is not None
            self.assertTrue(
                torch.allclose(
                    single_result.max_overlaps[0],
                    batched_result.max_overlaps[batch_index],
                    atol=1e-6,
                )
            )

    def test_assign_is_a_one_to_one_matching(self) -> None:
        """Test that no gt is claimed by two proposals and no proposal takes two gts."""
        result = self._assign()

        for batch_index in range(self.batch_size):
            positive_gt_inds = result.gt_inds[batch_index][result.gt_inds[batch_index] > 0]
            self.assertEqual(
                len(set(positive_gt_inds.tolist())),
                positive_gt_inds.numel(),
                msg=f"a gt was matched twice in batch element {batch_index}",
            )
            # At most one positive per real gt.
            self.assertLessEqual(positive_gt_inds.numel(), int(self.valid_masks[batch_index].sum()))

    def test_assign_never_matches_padded_gt(self) -> None:
        """Test that a padded gt column is never assigned, even with proposals to spare.

        Hungarian matching fills min(num_bboxes, max_num_gt_bboxes) pairs regardless of cost, so
        without an explicit filter a spare proposal would be paired with the padded column and
        gt_inds would point at padding.
        """
        result = self._assign()

        for batch_index in range(self.batch_size):
            gt_inds = result.gt_inds[batch_index]
            for proposal_index in range(self.num_bboxes):
                gt_index = int(gt_inds[proposal_index]) - 1
                if gt_index < 0:
                    continue
                self.assertTrue(
                    bool(self.valid_masks[batch_index, gt_index]),
                    msg=(
                        f"proposal {proposal_index} of batch element {batch_index} was matched "
                        f"to padded gt {gt_index}"
                    ),
                )

    def test_assign_negatives_carry_no_gt_or_label(self) -> None:
        """Test that unmatched proposals are negatives rather than ignored."""
        result = self._assign()

        negative_masks = result.gt_inds == 0
        self.assertTrue(negative_masks.any())
        self.assertTrue(torch.all(result.labels[negative_masks] == -1))
        assert result.max_overlaps is not None
        self.assertTrue(torch.all(result.max_overlaps[negative_masks] == 0.0))
        # -1 (ignore) is never produced by this assigner: every proposal is positive or negative.
        self.assertFalse(torch.any(result.gt_inds == -1))

    def test_assign_max_overlaps_are_the_matched_pair_ious(self) -> None:
        """Test that max_overlaps reports the IoU of the pair each proposal was matched to."""
        result = self._assign()
        ious = _bev_iou_aligned(self.bboxes, self.gt_bboxes, self.valid_masks)

        for batch_index in range(self.batch_size):
            for proposal_index in range(self.num_bboxes):
                gt_index = int(result.gt_inds[batch_index, proposal_index]) - 1
                expected_iou = (
                    0.0 if gt_index < 0 else float(ious[batch_index, proposal_index, gt_index])
                )
                assert result.max_overlaps is not None
                self.assertAlmostEqual(
                    float(result.max_overlaps[batch_index, proposal_index]),
                    expected_iou,
                    places=5,
                )
        # The proposals sitting on a gt overlap it almost exactly.
        assert result.max_overlaps is not None
        self.assertGreater(float(result.max_overlaps[0, 0]), 0.9)
        self.assertGreater(float(result.max_overlaps[1, 0]), 0.9)

    def test_assign_uses_classification_to_break_geometric_ties(self) -> None:
        """Test that with two geometrically identical proposals the classification cost decides."""
        # Both proposals sit exactly on the single gt, so only cls_pred separates them.
        bboxes = self._boxes_from_centers([[(2.0, 2.0), (2.0, 2.0)]])
        gt_bboxes = self._boxes_from_centers([[(2.0, 2.0)]])
        gt_labels = torch.tensor([[1]], dtype=torch.int64, device=self.device)
        valid_masks = torch.ones((1, 1), dtype=torch.bool, device=self.device)
        # Proposal 1 is the confident one about class 1.
        cls_pred = torch.tensor(
            [[[4.0, -4.0], [-4.0, 4.0]]], dtype=torch.float32, device=self.device
        )

        result = self.assigner.assign(
            bboxes=bboxes,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            cls_pred=cls_pred,
            valid_masks=valid_masks,
        )

        self.assertEqual(result.gt_inds[0].tolist(), [0, 1])
        self.assertEqual(result.labels[0].tolist(), [-1, 1])

    def test_assign_with_no_valid_gt_leaves_every_proposal_negative(self) -> None:
        """Test that a sample whose gts are all padded produces no positives."""
        valid_masks = torch.zeros_like(self.valid_masks)

        result = self.assigner.assign(
            bboxes=self.bboxes,
            gt_bboxes=self.gt_bboxes,
            gt_labels=self.gt_labels,
            cls_pred=self.cls_pred,
            valid_masks=valid_masks,
        )

        self.assertTrue(torch.all(result.gt_inds == 0))
        self.assertTrue(torch.all(result.labels == -1))
        # assign() short-circuits when no sample has a real gt and reports no overlaps at all,
        # which AssignResult allows (max_overlaps is Tensor | None and callers guard on it).
        if result.max_overlaps is not None:
            self.assertTrue(torch.all(result.max_overlaps == 0.0))
        self.assertEqual(result.num_gts.tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
