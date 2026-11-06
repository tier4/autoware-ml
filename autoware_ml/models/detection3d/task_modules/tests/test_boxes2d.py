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


"""Unit tests for the 2D box helpers of the auxiliary 2D head."""

from __future__ import annotations

import unittest

import torch
from torchvision.ops import box_iou, generalized_box_iou

from autoware_ml.models.detection3d.task_modules.boxes2d import (
    bbox_cxcywh_to_xyxy,
    bbox_overlaps,
    bbox_xyxy_to_cxcywh,
)


def _random_boxes(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    corner = torch.rand(count, 2, generator=generator) * 100.0
    size = torch.rand(count, 2, generator=generator) * 40.0 + 1.0
    return torch.cat([corner, corner + size], dim=-1)


class TestBoxFormats(unittest.TestCase):
    """Unit tests for the box form conversions."""

    def test_box_formats_convert_back_and_forth(self) -> None:
        """The center form of (10, 20) - (30, 60) is (20, 40, 20, 40); boxes survive a round trip."""
        box = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
        boxes = _random_boxes(16, seed=0)

        self.assertEqual(bbox_xyxy_to_cxcywh(box).tolist(), [[20.0, 40.0, 20.0, 40.0]])
        torch.testing.assert_close(bbox_cxcywh_to_xyxy(bbox_xyxy_to_cxcywh(boxes)), boxes)


class TestBBoxOverlaps(unittest.TestCase):
    """Unit tests for bbox_overlaps."""

    def test_overlaps_of_two_boxes_by_hand(self) -> None:
        """Two boxes sharing 1 of a union of 7, enclosed by 9, have IoU 1/7 and GIoU 1/7 - 2/9.

        A disjoint pair has IoU 0 and the negative GIoU 0 - (36 - 5) / 36.
        """
        first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        others = torch.tensor([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]])

        torch.testing.assert_close(bbox_overlaps(first, others), torch.tensor([[1 / 7, 0.0]]))
        torch.testing.assert_close(
            bbox_overlaps(first, others, mode="giou"), torch.tensor([[1 / 7 - 2 / 9, -31 / 36]])
        )

    def test_overlaps_match_torchvision(self) -> None:
        """Pairwise overlaps are torchvision's, and the aligned ones the diagonal of its result."""
        boxes1 = _random_boxes(12, seed=1)
        boxes2 = _random_boxes(9, seed=2)
        boxes3 = _random_boxes(12, seed=3)

        for mode, reference in (("iou", box_iou), ("giou", generalized_box_iou)):
            with self.subTest(mode=mode):
                torch.testing.assert_close(
                    bbox_overlaps(boxes1, boxes2, mode=mode), reference(boxes1, boxes2)
                )
                torch.testing.assert_close(
                    bbox_overlaps(boxes1, boxes3, mode=mode, is_aligned=True),
                    reference(boxes1, boxes3).diagonal(),
                )

    def test_identical_and_degenerate_boxes(self) -> None:
        """A box against itself gives 1; a zero-area box gives IoU 0 and GIoU -(enclose - area)."""
        box = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        point = torch.tensor([[6.0, 6.0, 6.0, 6.0]])

        torch.testing.assert_close(bbox_overlaps(box, box, mode="giou"), torch.tensor([[1.0]]))
        torch.testing.assert_close(bbox_overlaps(point, box), torch.tensor([[0.0]]))
        # The 6x6 enclosing box minus the 16 of the real box, over the enclosing 36.
        torch.testing.assert_close(
            bbox_overlaps(point, box, mode="giou"), torch.tensor([[-20 / 36]])
        )
        torch.testing.assert_close(bbox_overlaps(point, point, mode="giou"), torch.tensor([[0.0]]))

    def test_empty_inputs_give_empty_overlaps(self) -> None:
        self.assertEqual(bbox_overlaps(torch.zeros(0, 4), _random_boxes(3, seed=4)).shape, (0, 3))
        self.assertEqual(
            bbox_overlaps(_random_boxes(3, seed=4), torch.zeros(0, 4), mode="giou").shape, (3, 0)
        )

    def test_aligned_overlaps_reject_mismatched_counts(self) -> None:
        """Aligned mode pairs boxes one to one, so one box against three is an error."""
        with self.assertRaisesRegex(ValueError, "as many"):
            bbox_overlaps(_random_boxes(3, seed=5), _random_boxes(1, seed=6), is_aligned=True)

    def test_overlaps_reject_an_unknown_mode(self) -> None:
        box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        with self.assertRaisesRegex(ValueError, "distance"):
            bbox_overlaps(box, box, mode="distance")


if __name__ == "__main__":
    unittest.main()
