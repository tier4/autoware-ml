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

import torch

from autoware_ml.models.detection3d.task_modules.assigners2d import (
    AssignResult2D,
    HungarianAssigner2D,
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


def test_matches_every_ground_truth_box_to_the_proposal_on_top_of_it() -> None:
    """
    Input: three 20 x 20 px proposals of one image, two of them centered on the two ground-truth
    boxes and confident about their classes, the third away from both.
    Expected: the Hungarian assignment pairs every ground-truth box with the proposal on top of
    it and leaves the third proposal to the background.
    Check: the number of ground-truth boxes, and the 1-based ground-truth index (0 for the
    background) and the label (-1 for the background) of every proposal.
    """
    bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)])
    # Proposal 1 is confident about class 2, proposal 0 about class 0.
    cls_pred[1, 2] = 4.0
    cls_pred[0, 0] = 4.0
    gt_bboxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])
    gt_labels = torch.tensor([2, 0])

    result = HungarianAssigner2D().assign(
        bbox_pred,
        cls_pred,
        centers,
        gt_bboxes,
        gt_labels,
        gt_bboxes[:, :2] + 10.0,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    assert isinstance(result, AssignResult2D)
    assert result.num_gts == 2
    assert result.gt_inds.tolist() == [2, 1, 0]
    assert result.labels.tolist() == [0, 2, -1]


def test_an_image_without_ground_truth_assigns_every_proposal_to_background() -> None:
    """
    Input: two proposals of an image without ground-truth boxes.
    Expected: with nothing to match, every proposal is assigned to the background.
    Check: zero ground-truth boxes, and the ground-truth index 0 and the label -1 of every
    proposal.
    """
    bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0)])

    result = HungarianAssigner2D().assign(
        bbox_pred,
        cls_pred,
        centers,
        torch.zeros(0, 4),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, 2),
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    assert result.num_gts == 0
    assert result.gt_inds.tolist() == [0, 0]
    assert result.labels.tolist() == [-1, -1]


def test_bfloat16_logits_are_matched_like_their_float32_values() -> None:
    """Two proposals on top of two boxes of classes 0 and 1, so the classes decide the match.

    Autocast gives the logits in bfloat16, where the sigmoid of both rounds to 1 and the
    classification cost could no longer tell the proposals apart.
    """
    bbox_pred, _, centers = _proposals_at([(100.0, 50.0), (100.0, 50.0)], num_classes=2)
    # Proposal 0 is the more confident about class 1, proposal 1 about class 0.
    logits = torch.tensor([[7.0, 8.0], [8.0, 7.0]])
    gt_bboxes = torch.tensor([[90.0, 40.0, 110.0, 60.0], [90.0, 40.0, 110.0, 60.0]])

    for cls_pred in (logits, logits.bfloat16()):
        result = HungarianAssigner2D().assign(
            bbox_pred,
            cls_pred,
            centers,
            gt_bboxes,
            torch.tensor([0, 1]),
            gt_bboxes[:, :2] + 10.0,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )

        assert result.gt_inds.tolist() == [2, 1], cls_pred.dtype
