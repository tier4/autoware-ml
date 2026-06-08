"""Reusable proposal assigners for detection3d tasks.

This module contains assignment utilities shared by transformer-style 3D
detection heads during training target construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)


@dataclass
class AssignResult:
    """Store the output of proposal-to-ground-truth assignment.

    Attributes:
        num_gts: Number of ground-truth boxes used during matching.
        gt_inds: Assigned ground-truth indices for each proposal.
        max_overlaps: Optional overlap score for each assigned proposal.
        labels: Assigned class labels for each proposal.
    """

    num_gts: int
    gt_inds: torch.Tensor
    max_overlaps: torch.Tensor | None
    labels: torch.Tensor


def _bev_iou_aligned(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned BEV IoU for matching.

    Args:
        boxes: Proposal boxes in metric coordinates.
        gt_boxes: Ground-truth boxes in metric coordinates.

    Returns:
        Pairwise BEV IoU matrix between proposals and ground truth.
    """
    boxes_xy = boxes[:, :2]
    boxes_wh = boxes[:, 3:5].clamp_min(0)
    gt_xy = gt_boxes[:, :2]
    gt_wh = gt_boxes[:, 3:5].clamp_min(0)

    boxes_min = boxes_xy[:, None, :] - boxes_wh[:, None, :] * 0.5
    boxes_max = boxes_xy[:, None, :] + boxes_wh[:, None, :] * 0.5
    gt_min = gt_xy[None, :, :] - gt_wh[None, :, :] * 0.5
    gt_max = gt_xy[None, :, :] + gt_wh[None, :, :] * 0.5

    inter_min = torch.maximum(boxes_min, gt_min)
    inter_max = torch.minimum(boxes_max, gt_max)
    inter_wh = (inter_max - inter_min).clamp_min(0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    box_area = boxes_wh[:, 0] * boxes_wh[:, 1]
    gt_area = gt_wh[:, 0] * gt_wh[:, 1]
    union = box_area[:, None] + gt_area[None, :] - inter_area
    return inter_area / union.clamp_min(1e-6)


@dataclass
class HungarianAssigner3D:
    """Assign proposals to targets with weighted Hungarian matching.

    Attributes:
        cls_cost: Classification cost term.
        reg_cost: Bounding-box regression cost term.
        iou_cost: IoU-based matching cost term.
    """

    cls_cost: ClassificationCost
    reg_cost: BBoxBEVL1Cost
    iou_cost: IoU3DCost

    def assign(
        self,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        cls_pred: torch.Tensor,
        point_cloud_range: list[float],
    ) -> AssignResult:
        """Assign proposals to ground truth using weighted Hungarian matching.

        Args:
            bboxes: Proposal boxes.
            gt_bboxes: Ground-truth boxes.
            gt_labels: Ground-truth class labels.
            cls_pred: Classification predictions associated with ``bboxes`` in
                ``(num_classes, num_bboxes)`` layout. The assigner transposes
                this tensor before evaluating classification cost.
            point_cloud_range: Detector point-cloud range used by the regression cost.

        Returns:
            Assignment result with matched indices, labels, and overlaps.

        """
        num_gts, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts, gt_inds=assigned_gt_inds, max_overlaps=None, labels=assigned_labels
            )

        cls_cost = self.cls_cost(cls_pred.transpose(0, 1), gt_labels)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, point_cloud_range)
        iou = _bev_iou_aligned(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)
        cost = (
            torch.nan_to_num(cls_cost + reg_cost + iou_cost, nan=1e6, posinf=1e6, neginf=-1e6)
            .detach()
            .cpu()
        )

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        max_overlaps = torch.zeros(num_bboxes, device=bboxes.device, dtype=bboxes.dtype)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels,
        )
