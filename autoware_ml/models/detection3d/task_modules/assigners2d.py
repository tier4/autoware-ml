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

"""Hungarian assignment for the auxiliary 2D detection head.

The assigner matches dense image-plane proposals against projected 2D ground
truth with the DETR-style weighted cost: classification, box L1, GIoU, and
projected-center L1 (the Focal-PETR extension).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from scipy.optimize import linear_sum_assignment

from autoware_ml.models.detection3d.task_modules.assigners import AssignResult
from autoware_ml.models.detection3d.task_modules.boxes2d import (
    bbox_cxcywh_to_xyxy,
    bbox_overlaps,
    bbox_xyxy_to_cxcywh,
)
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost


@dataclass(frozen=True)
class BBoxL1Cost2D:
    """Pairwise L1 cost between normalized 2D boxes.

    Predictions are ``(cx, cy, w, h)`` in [0, 1]; ground truth arrives as
    normalized ``(x1, y1, x2, y2)`` and is converted according to
    ``box_format``.
    """

    weight: float = 1.0
    box_format: str = "xywh"

    def __call__(self, bbox_pred: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise box L1 cost."""
        if self.box_format == "xywh":
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == "xyxy":
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        else:
            raise ValueError(f"Unsupported box_format '{self.box_format}'.")
        return torch.cdist(bbox_pred.float(), gt_bboxes.float(), p=1) * self.weight


@dataclass(frozen=True)
class IoUCost2D:
    """Pairwise negative-IoU (or GIoU) cost between unnormalized 2D boxes."""

    weight: float = 1.0
    iou_mode: str = "giou"

    def __call__(self, bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise overlap cost."""
        overlaps = bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        return -overlaps * self.weight


@dataclass(frozen=True)
class Center2DL1Cost:
    """Pairwise L1 cost between normalized projected 2D centers."""

    weight: float = 1.0

    def __call__(self, center_pred: torch.Tensor, gt_centers: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise center L1 cost."""
        return torch.cdist(center_pred.float(), gt_centers.float(), p=1) * self.weight


@dataclass
class HungarianAssigner2D:
    """One-to-one assignment between dense 2D proposals and ground truth.

    The default cost weights mirror the reference Focal-PETR recipe
    (2 / 5 / 2 / 10), matching :class:`FocalHead2D`'s default loss weights.
    """

    cls_cost: ClassificationCost = field(default_factory=lambda: ClassificationCost(weight=2.0))
    reg_cost: BBoxL1Cost2D = field(default_factory=lambda: BBoxL1Cost2D(weight=5.0))
    iou_cost: IoUCost2D = field(default_factory=lambda: IoUCost2D(weight=2.0))
    centers2d_cost: Center2DL1Cost = field(default_factory=lambda: Center2DL1Cost(weight=10.0))

    def assign(
        self,
        bbox_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        pred_centers2d: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_centers2d: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> AssignResult:
        """Assign each proposal to background (0) or a 1-based ground-truth index.

        Args:
            bbox_pred: Proposal boxes ``(num_queries, 4)`` as normalized
                ``(cx, cy, w, h)``.
            cls_pred: Proposal classification logits ``(num_queries, C)``.
            pred_centers2d: Proposal centers ``(num_queries, 2)`` normalized.
            gt_bboxes: Ground-truth boxes ``(num_gts, 4)`` in unnormalized
                ``(x1, y1, x2, y2)`` pixels.
            gt_labels: Ground-truth labels ``(num_gts,)``.
            gt_centers2d: Ground-truth projected centers ``(num_gts, 2)`` in
                unnormalized pixels.
            image_height: Padded image height in pixels.
            image_width: Padded image width in pixels.

        Returns:
            Assignment result over the proposals.
        """
        num_gts = gt_bboxes.size(0)
        num_queries = bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_zeros((num_queries,), dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_queries,), -1, dtype=torch.long)
        if num_gts == 0 or num_queries == 0:
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels,
            )

        factor = gt_bboxes.new_tensor([image_width, image_height, image_width, image_height])
        cost = (
            self.cls_cost(cls_pred, gt_labels)
            + self.reg_cost(bbox_pred, gt_bboxes / factor)
            + self.iou_cost(bbox_cxcywh_to_xyxy(bbox_pred) * factor, gt_bboxes)
            + self.centers2d_cost(pred_centers2d, gt_centers2d / factor[:2])
        )
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_rows, matched_cols = linear_sum_assignment(cost.detach().cpu().numpy())
        matched_rows = torch.as_tensor(matched_rows, device=bbox_pred.device, dtype=torch.long)
        matched_cols = torch.as_tensor(matched_cols, device=bbox_pred.device, dtype=torch.long)
        assigned_gt_inds[matched_rows] = matched_cols + 1
        assigned_labels[matched_rows] = gt_labels[matched_cols].long()
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels,
        )
