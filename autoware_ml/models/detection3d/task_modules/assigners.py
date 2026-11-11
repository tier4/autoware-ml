"""Reusable proposal assigners for detection3d tasks.

This module contains assignment utilities shared by transformer-style 3D
detection heads during training target construction, and the per-image 2D
assigner of the auxiliary image-plane head (Focal-PETR): classification, box
L1, GIoU and projected-center L1 costs.
"""

from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass, field

from jaxtyping import Float32, Bool, Int32, Int64
import torch
from scipy.optimize import linear_sum_assignment

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    BBoxBEVL1Cost,
    BBoxL1Cost2D,
    Center2DL1Cost,
    ClassificationCost,
    IoU3DCost,
    IoUCost2D,
)
from autoware_ml.types.geometry import Box3DFieldIndex


def _linear_sum_assignment(
    cost: Float32[torch.Tensor, "num_proposals num_gts"],
) -> tuple[Int64[torch.Tensor, " num_matches"], Int64[torch.Tensor, " num_matches"]]:
    """Run SciPy's Hungarian solver on one cost block.

    Args:
        cost: Pairwise cost block ``(num_proposals, num_gts)`` of finite values.

    Returns:
        The matched proposal rows and ground-truth columns, as long tensors on the device of
        ``cost``.
    """
    matched_rows, matched_cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(matched_rows, device=cost.device, dtype=torch.long),
        torch.as_tensor(matched_cols, device=cost.device, dtype=torch.long),
    )


@dataclass(frozen=True)
class AssignResult:
    """Store the output of proposal-to-ground-truth assignment.

    Attributes:
        num_gts: Number of ground-truth boxes used during matching.
        gt_inds: Assigned ground-truth indices for each proposal, offset by one.
            Unassigned proposals carry 0.
        labels: Assigned class labels for each proposal.
        max_overlaps: Optional overlap score for each assigned proposal.
    """

    num_gts: Int32[torch.Tensor, " batch_size"]
    gt_inds: Int64[torch.Tensor, "batch_size num_bboxes"]
    labels: Int64[torch.Tensor, "batch_size num_bboxes"]
    max_overlaps: Float32[torch.Tensor, "batch_size num_bboxes"] | None


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
        reg_cost: Bounding-box regression cost term, called with the whole boxes either way.
            ``BBoxBEVL1Cost`` reads only their BEV centers, ``BBox3DL1Cost`` the whole box
            encoding.
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
        gt_labels: Int64[torch.Tensor, "batch_size max_num_gt_bboxes"],
        cls_pred: Float32[torch.Tensor, "batch_size num_bboxes num_classes"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
    ) -> AssignResult:
        """Assign proposals to ground truth using weighted Hungarian matching.

        Args:
            bboxes: Proposal boxes, where code_size is
                (center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y).
            gt_bboxes: Ground-truth boxes, where code_size is
                (center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y).
            gt_labels: Ground-truth class labels, -1 for the padded gt boxes.
            cls_pred: Classification logits of every proposal, one row per box in the same
                order as ``bboxes`` and one column per class.
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
            bboxes=bboxes,
            gt_bboxes=gt_bboxes,
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

        # linear_sum_assignment takes one cost matrix at a time, so loop over the batch.
        batch_size, num_bboxes = bboxes.shape[0], bboxes.shape[1]
        assigned_gt_inds = bboxes.new_full((batch_size, num_bboxes), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((batch_size, num_bboxes), -1, dtype=torch.long)
        max_overlaps = torch.zeros(
            (batch_size, num_bboxes), device=bboxes.device, dtype=bboxes.dtype
        )
        for current_batch in range(batch_size):
            matched_row_inds, matched_col_inds = _linear_sum_assignment(cost[current_batch])
            matched_row_inds = matched_row_inds.to(bboxes.device)
            matched_col_inds = matched_col_inds.to(bboxes.device)

            # Hungarian matching always fills min(num_bboxes, max_num_gt_bboxes) pairs, so it
            # matches the padded gt columns too even though their cost is the maximum. Drop those
            # pairs, otherwise gt_inds would point at padding and the caller's
            # `gt_inds[gt_inds > 0] - 1` would gather padded boxes as regression targets.
            valid_pairs = valid_masks[current_batch, matched_col_inds]
            matched_row_inds = matched_row_inds[valid_pairs]
            matched_col_inds = matched_col_inds[valid_pairs]

            # Every proposal starts as a negative (0). Only matched ones carry a gt index.
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


@dataclass(frozen=True)
class AssignResult2D:
    """Assignment of the dense proposals of one image to its ground truth.

    Attributes:
        num_gts: Number of ground-truth boxes of the image.
        gt_inds: Assigned ground-truth index of every proposal, offset by one; background
            proposals carry 0.
        labels: Assigned class label of every proposal, -1 for background.
    """

    num_gts: int
    gt_inds: Int64[torch.Tensor, " num_queries"]
    labels: Int64[torch.Tensor, " num_queries"]


@dataclass(frozen=True)
class HungarianAssigner2D:
    """One-to-one assignment between dense 2D proposals and ground truth.

    The default cost weights are the reference Focal-PETR recipe (2 / 5 / 2 / 10), the same
    values the auxiliary 2D head uses as its loss weights.
    """

    cls_cost: ClassificationCost = field(default_factory=lambda: ClassificationCost(weight=2.0))
    reg_cost: BBoxL1Cost2D = field(default_factory=lambda: BBoxL1Cost2D(weight=5.0))
    iou_cost: IoUCost2D = field(default_factory=lambda: IoUCost2D(weight=2.0))
    centers2d_cost: Center2DL1Cost = field(default_factory=lambda: Center2DL1Cost(weight=10.0))

    def assign(
        self,
        bbox_pred: Float32[torch.Tensor, "num_queries 4"],
        cls_pred: Float32[torch.Tensor, "num_queries num_classes"],
        pred_centers2d: Float32[torch.Tensor, "num_queries 2"],
        gt_bboxes: Float32[torch.Tensor, "num_gts 4"],
        gt_labels: Int64[torch.Tensor, " num_gts"],
        gt_centers2d: Float32[torch.Tensor, "num_gts 2"],
        image_height: int,
        image_width: int,
    ) -> AssignResult2D:
        """Assign each proposal to background (0) or a 1-based ground-truth index.

        Args:
            bbox_pred: Proposal boxes ``(num_queries, 4)`` as normalized
                ``(cx, cy, w, h)``.
            cls_pred: Proposal classification logits ``(num_queries, C)``.
            pred_centers2d: Proposal centers ``(num_queries, 2)`` normalized.
            gt_bboxes: Ground-truth boxes ``(num_gts, 4)`` in unnormalized
                ``(x1, y1, x2, y2)`` pixels.
            gt_labels: Ground-truth class indices ``(num_gts,)``. The ground truth of one image
                is not padded, so every label must be a class index; -1 is not a padding
                sentinel here.
            gt_centers2d: Ground-truth projected centers ``(num_gts, 2)`` in
                unnormalized pixels.
            image_height: Padded image height in pixels.
            image_width: Padded image width in pixels.

        Returns:
            Assignment result over the proposals.

        Raises:
            ValueError: If a box, logit or center tensor is not float32, or a label is negative
                or not below the number of class channels.
        """
        # Match in float32 like the reference (force_fp32): a bfloat16 sigmoid saturates above
        # logit 6.2, and a bfloat16 box is off by up to 1/256 of the image. Callers running
        # under autocast cast their outputs first.
        inputs = {
            "bbox_pred": bbox_pred,
            "cls_pred": cls_pred,
            "pred_centers2d": pred_centers2d,
            "gt_bboxes": gt_bboxes,
            "gt_centers2d": gt_centers2d,
        }
        not_float32 = {
            name: tensor.dtype for name, tensor in inputs.items() if tensor.dtype != torch.float32
        }
        if not_float32:
            raise ValueError(f"HungarianAssigner2D matches in float32, got {not_float32}.")

        num_gts = gt_bboxes.size(0)
        num_queries = bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_zeros((num_queries,), dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_queries,), -1, dtype=torch.long)
        if num_gts == 0 or num_queries == 0:
            return AssignResult2D(num_gts=num_gts, gt_inds=assigned_gt_inds, labels=assigned_labels)
        # ClassificationCost treats -1 as padding; this assigner drops no pair, so such a box
        # would still be matched. The 2D ground truth is not padded.
        if bool((gt_labels < 0).any()):
            raise ValueError("Found negative gt_labels. The 2D ground truth is not padded.")

        factor = gt_bboxes.new_tensor([image_width, image_height, image_width, image_height])
        # ClassificationCost is batched; one image is a batch of one with all boxes valid.
        valid_masks = torch.ones((1, num_gts), dtype=torch.bool, device=gt_labels.device)
        cls_cost = self.cls_cost(cls_pred.unsqueeze(0), gt_labels.unsqueeze(0), valid_masks)[0]
        cost = (
            cls_cost
            + self.reg_cost(bbox_pred, gt_bboxes / factor)
            + self.iou_cost(BaseBBoxes2D.from_cxcywh(bbox_pred).bbox_params * factor, gt_bboxes)
            + self.centers2d_cost(pred_centers2d, gt_centers2d / factor[:2])
        )
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_rows, matched_cols = _linear_sum_assignment(cost)
        assigned_gt_inds[matched_rows] = matched_cols + 1
        assigned_labels[matched_rows] = gt_labels[matched_cols].long()
        return AssignResult2D(num_gts=num_gts, gt_inds=assigned_gt_inds, labels=assigned_labels)
