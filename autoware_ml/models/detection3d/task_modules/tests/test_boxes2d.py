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
    """The center form of (10, 20) - (30, 60) is (20, 40, 20, 40); boxes survive a round trip."""
    box = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    boxes = _random_boxes(16, seed=0)

    assert bbox_xyxy_to_cxcywh(box).tolist() == [[20.0, 40.0, 20.0, 40.0]]
    torch.testing.assert_close(bbox_cxcywh_to_xyxy(bbox_xyxy_to_cxcywh(boxes)), boxes)


def test_overlaps_of_two_boxes_by_hand() -> None:
    """Two boxes sharing 1 of a union of 7, enclosed by 9, have IoU 1/7 and GIoU 1/7 - 2/9.

    A disjoint pair has IoU 0 and the negative GIoU 0 - (36 - 5) / 36.
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
    """Pairwise overlaps are torchvision's, and the aligned ones the diagonal of its result."""
    boxes1 = _random_boxes(12, seed=1)
    boxes2 = _random_boxes(9, seed=2)
    boxes3 = _random_boxes(12, seed=3)

    torch.testing.assert_close(bbox_overlaps(boxes1, boxes2, mode=mode), reference(boxes1, boxes2))
    torch.testing.assert_close(
        bbox_overlaps(boxes1, boxes3, mode=mode, is_aligned=True),
        reference(boxes1, boxes3).diagonal(),
    )


def test_overlaps_reject_an_unknown_mode() -> None:
    """Only "iou" and "giou" are supported."""
    box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="distance"):
        bbox_overlaps(box, box, mode="distance")
