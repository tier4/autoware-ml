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


"""Unit tests for BaseBBoxes2D."""

from __future__ import annotations

import unittest

import torch
from torchvision.ops import box_iou, generalized_box_iou

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D


class TestBaseBBoxes2D(unittest.TestCase):
    """Unit tests for BaseBBoxes2D."""

    def setUp(self) -> None:
        """Build three sets of random pixel boxes, of 12, 9 and 12 boxes."""
        generator = torch.Generator().manual_seed(0)
        self.random_boxes = []
        for count in (12, 9, 12):
            corner = torch.rand(count, 2, generator=generator) * 100.0
            size = torch.rand(count, 2, generator=generator) * 40.0 + 1.0
            self.random_boxes.append(BaseBBoxes2D(torch.cat([corner, corner + size], dim=-1)))

    def test_box_formats_convert_back_and_forth(self) -> None:
        """
        Input: the box (10, 20) - (30, 60), and 12 random boxes.
        Expected: the center form of the box is (20, 40, 20, 40), its area 800, and every box
        survives a round trip through the center form.
        Check: the center form and the area by hand, and the round trip.
        """
        box = BaseBBoxes2D(torch.tensor([[10.0, 20.0, 30.0, 60.0]]))
        boxes = self.random_boxes[0]

        self.assertEqual(box.cxcywh.tolist(), [[20.0, 40.0, 20.0, 40.0]])
        self.assertEqual(box.area.tolist(), [800.0])
        torch.testing.assert_close(
            BaseBBoxes2D.from_cxcywh(boxes.cxcywh).bbox_params, boxes.bbox_params
        )

    def test_overlaps_of_two_boxes_by_hand(self) -> None:
        """
        Input: (0, 0) - (2, 2) against (1, 1) - (3, 3) and the disjoint (5, 5) - (6, 6).
        Expected: the first pair shares 1 of a union of 7 inside an enclosing box of 9, so its
        IoU is 1/7 and its GIoU 1/7 - 2/9. The disjoint pair has IoU 0 and GIoU
        0 - (36 - 5) / 36.
        Check: both overlaps against these values.
        """
        first = BaseBBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0]]))
        others = BaseBBoxes2D(torch.tensor([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]]))

        torch.testing.assert_close(first.iou(others), torch.tensor([[1 / 7, 0.0]]))
        torch.testing.assert_close(first.giou(others), torch.tensor([[1 / 7 - 2 / 9, -31 / 36]]))

    def test_overlaps_match_torchvision(self) -> None:
        """
        Input: random boxes, 12 against 9 pairwise and 12 against 12 aligned.
        Expected: the pairwise overlaps are torchvision's, and the aligned ones the diagonal of
        its result.
        Check: IoU against box_iou and GIoU against generalized_box_iou.
        """
        boxes1, boxes2, boxes3 = self.random_boxes
        for name, reference in (("iou", box_iou), ("giou", generalized_box_iou)):
            with self.subTest(name):
                overlaps = getattr(boxes1, name)
                torch.testing.assert_close(
                    overlaps(boxes2), reference(boxes1.bbox_params, boxes2.bbox_params)
                )
                torch.testing.assert_close(
                    overlaps(boxes3, is_aligned=True),
                    reference(boxes1.bbox_params, boxes3.bbox_params).diagonal(),
                )

    def test_identical_and_degenerate_boxes(self) -> None:
        """
        Input: the box (0, 0) - (4, 4) and the zero-area box (6, 6) - (6, 6).
        Expected: a box against itself gives 1. The zero-area box gives IoU 0, GIoU minus the
        share of the 6 x 6 enclosing box outside the real one, -20/36, and GIoU 0 against
        itself instead of NaN.
        Check: the overlaps against these values.
        """
        box = BaseBBoxes2D(torch.tensor([[0.0, 0.0, 4.0, 4.0]]))
        point = BaseBBoxes2D(torch.tensor([[6.0, 6.0, 6.0, 6.0]]))

        torch.testing.assert_close(box.giou(box), torch.tensor([[1.0]]))
        torch.testing.assert_close(point.iou(box), torch.tensor([[0.0]]))
        torch.testing.assert_close(point.giou(box), torch.tensor([[-20 / 36]]))
        torch.testing.assert_close(point.giou(point), torch.tensor([[0.0]]))

    def test_empty_inputs_give_empty_overlaps(self) -> None:
        """
        Input: no boxes against three, and three against none.
        Expected: overlaps with no rows or no columns.
        Check: the shapes (0, 3) and (3, 0).
        """
        empty = BaseBBoxes2D(torch.zeros(0, 4))
        boxes = BaseBBoxes2D(self.random_boxes[1].bbox_params[:3])

        self.assertEqual(empty.iou(boxes).shape, (0, 3))
        self.assertEqual(boxes.giou(empty).shape, (3, 0))

    def test_boxes_must_be_float32_and_two_dimensional(self) -> None:
        """
        Input: float16 and bfloat16 boxes, built directly and from the center form, and a
        (batch, num_boxes, 4) tensor.
        Expected: half precision overflows the area of a pixel box (a 1900 x 1000 box is 1.9e6,
        above the float16 maximum of 65504), and batched boxes would broadcast into a wrong
        shape, so every one is refused when it is built.
        Check: the ValueError naming float32 or the expected shape.
        """
        boxes = torch.tensor([[0.0, 0.0, 1900.0, 1000.0]])
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "float32"):
                    BaseBBoxes2D(boxes.to(dtype))
                with self.assertRaisesRegex(ValueError, "float32"):
                    BaseBBoxes2D.from_cxcywh(boxes.to(dtype))
        with self.assertRaisesRegex(ValueError, r"\(num_bboxes, 4\)"):
            BaseBBoxes2D(torch.rand(2, 5, 4))

    def test_aligned_overlaps_reject_mismatched_counts(self) -> None:
        """
        Input: three boxes against one, aligned.
        Expected: aligned overlaps pair the boxes one to one, so the counts must match.
        Check: the ValueError of both overlaps.
        """
        three = BaseBBoxes2D(self.random_boxes[0].bbox_params[:3])
        one = BaseBBoxes2D(self.random_boxes[1].bbox_params[:1])

        for name in ("iou", "giou"):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "as many"):
                    getattr(three, name)(one, is_aligned=True)


if __name__ == "__main__":
    unittest.main()
