"""Reusable proposal assigners for detection3d tasks.

This module contains assignment utilities shared by transformer-style 3D
detection heads during training target construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from jaxtyping import Float32, Bool, Int32, Int64
import torch
from scipy.optimize import linear_sum_assignment

from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.types.geometry import Box3DFieldIndex


@dataclass(frozen=True)
class AssignResult:
    """Store the output of proposal-to-ground-truth assignment.

    Attributes:
        num_gts: Number of ground-truth boxes used during matching.
        gt_inds: Assigned ground-truth indices for each proposal.
            For unassigned est bboxes, they are 0.
        labels: Assigned class labels for each proposal.
        max_overlaps: Optional overlap score for each assigned proposal.
    """

    num_gts: Int32[torch.Tensor, " batch_size"]
    gt_inds: Int64[torch.Tensor, "batch_size num_bboxes"]
    labels: Int64[torch.Tensor, "batch_size num_"]
    max_overlaps: Float32[torch.Tensor, " batch_size num_bboxes"] | None


def _bev_iou_aligned(
    boxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
    gt_boxes: Float32[torch.Tensor, "batch_size max_num_gt_bboxes code_size"],
    valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
) -> Float32[torch.Tensor, "batch_size num_bboxes max_num_gt_bboxes"]:
    """Compute axis-aligned BEV IoU for matching. For invalid gt bboxes, it returns 0.

    The IoU is axis aligned, so yaw is ignored: LENGTH is the extent along x and WIDTH the extent
    along y. Like the matching costs, this is a batch-wise implementation that builds the full
    pairwise block per batch element and then zeroes the columns of invalid gt bboxes so they
    never look like a good match.

    Args:
        boxes: Proposal boxes in metric coordinates.
        gt_boxes: Ground-truth boxes in metric coordinates.
        valid_masks: Boolean mask indicating which gt bboxes are valid.

    Returns:
        Pairwise BEV IoU matrix between proposals and ground truth, one
        (num_bboxes, num_gt_bboxes) block per batch element.
    """
    # Slice the field axis, which is the trailing one. (batch_size, num_bboxes, 2)
    boxes_xy = boxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Y + 1]
    boxes_wh = boxes[..., Box3DFieldIndex.LENGTH : Box3DFieldIndex.WIDTH + 1].clamp_min(0)
    # (batch_size, num_gt_bboxes, 2)
    gt_xy = gt_boxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Y + 1]
    gt_wh = gt_boxes[..., Box3DFieldIndex.LENGTH : Box3DFieldIndex.WIDTH + 1].clamp_min(0)

    # Corners of every proposal and gt box, laid out so they broadcast into the pairwise grid.
    # (batch_size, num_bboxes, 1, 2)
    boxes_min = (boxes_xy - boxes_wh * 0.5).unsqueeze(2)
    boxes_max = (boxes_xy + boxes_wh * 0.5).unsqueeze(2)
    # (batch_size, 1, num_gt_bboxes, 2)
    gt_min = (gt_xy - gt_wh * 0.5).unsqueeze(1)
    gt_max = (gt_xy + gt_wh * 0.5).unsqueeze(1)

    # (batch_size, num_bboxes, num_gt_bboxes, 2)
    inter_min = torch.maximum(boxes_min, gt_min)
    inter_max = torch.minimum(boxes_max, gt_max)
    inter_wh = (inter_max - inter_min).clamp_min(0)
    # (batch_size, num_bboxes, num_gt_bboxes)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # (batch_size, num_bboxes) and (batch_size, num_gt_bboxes)
    box_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    gt_area = gt_wh[..., 0] * gt_wh[..., 1]
    # (batch_size, num_bboxes, 1) + (batch_size, 1, num_gt_bboxes) - (batch_size, num_bboxes,
    # num_gt_bboxes) -> (batch_size, num_bboxes, num_gt_bboxes)
    union = box_area.unsqueeze(2) + gt_area.unsqueeze(1) - inter_area
    ious = inter_area / union.clamp_min(1e-6)
    # Invalid gt boxes always return lowest iou
    # valid_masks: (batch_size, num_gt_bboxes) -> (batch_size, 1, num_gt_bboxes)
    # Broadcast to every column (gt_bboxes)
    return ious.masked_fill(~valid_masks.unsqueeze(1), 0.0)


@dataclass(frozen=True)
class HungarianAssigner3D:
    """Assign proposals to targets with weighted Hungarian matching.

    Attributes:
        cls_cost: Classification cost term.
        reg_cost: Bounding-box regression cost term.
        iou_cost: IoU-based matching cost term.
        point_cloud_range: Detector point-cloud range used by the regression cost.
    """

    cls_cost: ClassificationCost
    reg_cost: BBoxBEVL1Cost | BBox3DL1Cost
    iou_cost: IoU3DCost
    point_cloud_range: Sequence[float]

    def assign(
        self,
        bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        gt_bboxes: Float32[torch.Tensor, "batch_size max_num_gt_bboxes code_size"],
        gt_labels: Float32[torch.Tensor, "batch_size max_num_gt_bboxes"],
        cls_pred: Float32[torch.Tensor, "batch_size num_bboxes num_classes"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
    ) -> AssignResult:
        """Assign proposals to ground truth using weighted Hungarian matching.

        Args:
            bboxes: Proposal boxes, where code_size is
                (center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y).
            gt_bboxes: Ground-truth boxes, where code_size is
                (center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y).
            gt_labels: Ground-truth class labels.
            cls_pred: Classification predictions associated with ``bboxes`` in
                ``(num_classes, num_bboxes)`` layout. The assigner transposes
                this tensor before evaluating classification cost.
            valid_masks: Boolean mask to indicate if a gt box is valid.

        Returns:
            Assignment result with matched indices, labels, and overlaps.

        """
        # Check if any empty bboxes
        # (batch_size,) number of real gt boxes per sample, ignoring the padded columns.
        num_gts = valid_masks.sum(dim=1).to(torch.int32)
        batch_size, num_bboxes, _ = bboxes.shape

        assigned_gt_inds = bboxes.new_full((batch_size, num_bboxes), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((batch_size, num_bboxes), -1, dtype=torch.long)
        max_num_gts = torch.max(num_gts)
        if max_num_gts == 0 or num_bboxes == 0:
            if max_num_gts == 0:
                assigned_gt_inds[:] = 0
                num_gts = torch.zeros(batch_size, device=bboxes.device, dtype=torch.int32)

            return AssignResult(
                num_gts=num_gts, gt_inds=assigned_gt_inds, max_overlaps=None, labels=assigned_labels
            )

        cls_cost = self.cls_cost(gt_labels=gt_labels, cls_logits=cls_pred, valid_masks=valid_masks)
        reg_cost = self.reg_cost(
            bboxes_centers=bboxes[:, :, Box3DFieldIndex.X : Box3DFieldIndex.Y + 1],
            gt_bboxes_centers=gt_bboxes[:, :, Box3DFieldIndex.X : Box3DFieldIndex.Y + 1],
            valid_masks=valid_masks,
            point_cloud_range=self.point_cloud_range,
        )

        iou = _bev_iou_aligned(bboxes, gt_bboxes, valid_masks=valid_masks)
        iou_cost = self.iou_cost(iou)
        # (batch_size, num_bboxes, num_gt_bboxes)
        cost = (
            torch.nan_to_num(cls_cost + reg_cost + iou_cost, nan=1e6, posinf=1e6, neginf=-1e6)
            .detach()
            .cpu()
        )

        # linear sum assignment only works in sequantially
        batch_size, num_bboxes = bboxes.shape[0], bboxes.shape[1]
        assigned_gt_inds = bboxes.new_full((batch_size, num_bboxes), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((batch_size, num_bboxes), -1, dtype=torch.long)
        max_overlaps = torch.zeros(
            (batch_size, num_bboxes), device=bboxes.device, dtype=bboxes.dtype
        )
        for current_batch in range(batch_size):
            current_cost = cost[current_batch]
            matched_row_inds, matched_col_inds = linear_sum_assignment(current_cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

            # Hungarian matching always fills min(num_bboxes, max_num_gt_bboxes) pairs, so it
            # matches the padded gt columns too even though their cost is the maximum. Drop those
            # pairs, otherwise gt_inds would point at padding and the caller's
            # `gt_inds[gt_inds > 0] - 1` would gather padded boxes as regression targets.
            valid_pairs = valid_masks[current_batch, matched_col_inds]
            matched_row_inds = matched_row_inds[valid_pairs]
            matched_col_inds = matched_col_inds[valid_pairs]

            # Every proposal starts as a negative (0); only matched ones carry a gt index.
            assigned_gt_inds[current_batch, :] = 0
            assigned_gt_inds[current_batch, matched_row_inds] = matched_col_inds + 1
            assigned_labels[current_batch, matched_row_inds] = gt_labels[
                current_batch, matched_col_inds
            ]
            # iou is batched, so the sample axis has to be indexed too.
            max_overlaps[current_batch, matched_row_inds] = iou[
                current_batch, matched_row_inds, matched_col_inds
            ]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels,
        )
