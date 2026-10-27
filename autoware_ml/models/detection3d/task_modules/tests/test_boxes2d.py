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

import pytest
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


def test_box_formats_convert_back_and_forth() -> None:
    """
    Input: the box (10, 20) - (30, 60) in corner form, and random boxes.
    Expected: the center form of the box is (20, 40, 20, 40), and converting any box to the
    center form and back gives the box again.
    Check: the converted box, and the round trip of the random boxes.
    """
    box = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    boxes = _random_boxes(16, seed=0)

    assert bbox_xyxy_to_cxcywh(box).tolist() == [[20.0, 40.0, 20.0, 40.0]]
    torch.testing.assert_close(bbox_cxcywh_to_xyxy(bbox_xyxy_to_cxcywh(boxes)), boxes)


def test_overlaps_of_two_boxes_by_hand() -> None:
    """
    Input: the boxes (0, 0) - (2, 2) and (1, 1) - (3, 3), and a third box (5, 5) - (6, 6) apart
    from the first.
    Expected: the first two share an area of 1 over a union of 7, inside an enclosing box of 9,
    so their IoU is 1/7 and their GIoU 1/7 - 2/9. The first and the third do not overlap, so
    their IoU is 0 and their GIoU is negative: 0 - (36 - 5) / 36.
    Check: the IoU and GIoU of both pairs.
    """
    first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    others = torch.tensor([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]])

    torch.testing.assert_close(bbox_overlaps(first, others), torch.tensor([[1 / 7, 0.0]]))
    torch.testing.assert_close(
        bbox_overlaps(first, others, mode="giou"), torch.tensor([[1 / 7 - 2 / 9, -31 / 36]])
    )


@pytest.mark.parametrize(
    "mode, reference", [("iou", box_iou), ("giou", generalized_box_iou)], ids=["iou", "giou"]
)
def test_overlaps_match_torchvision(mode: str, reference) -> None:
    """
    Input: 12 and 9 random boxes, compared pairwise, and 12 against 12 element by element.
    Expected: the pairwise overlaps are those of torchvision, and the aligned overlaps are the
    diagonal of the pairwise ones.
    Check: the (12, 9) pairwise overlaps against torchvision, and the (12,) aligned overlaps
    against the diagonal of torchvision's (12, 12) result.
    """
    boxes1 = _random_boxes(12, seed=1)
    boxes2 = _random_boxes(9, seed=2)
    boxes3 = _random_boxes(12, seed=3)

    torch.testing.assert_close(bbox_overlaps(boxes1, boxes2, mode=mode), reference(boxes1, boxes2))
    torch.testing.assert_close(
        bbox_overlaps(boxes1, boxes3, mode=mode, is_aligned=True),
        reference(boxes1, boxes3).diagonal(),
    )


def test_overlaps_reject_an_unknown_mode() -> None:
    """
    Input: the overlap mode "distance", which is not implemented.
    Expected: only IoU and GIoU are supported, so the mode is refused.
    Check: a ValueError naming the mode is raised.
    """
    box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="distance"):
        bbox_overlaps(box, box, mode="distance")
