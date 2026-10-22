"""Reusable matching costs for detection3d assignment.

This module contains pairwise matching costs used by Hungarian assignment.
"""

from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass

from jaxtyping import Float32, Int64, Bool
import torch

from autoware_ml.models.detection3d.task_modules.bbox_coders import normalize_boxes3d
from autoware_ml.types.geometry import Box3DFieldIndex


@dataclass(frozen=True)
class ClassificationCost:
    """Compute focal-style classification cost for Hungarian matching.

    The cost mirrors the classification term used by query-based detection
    assigners.
    """

    weight: float = 1.0
    alpha: float = 0.25
    gamma: float = 2.0
    eps: float = 1e-12

    def __call__(
        self,
        cls_logits: Float32[torch.Tensor, "batch_size num_bboxes num_classes"],
        gt_labels: Int64[torch.Tensor, "batch_size max_num_gt_bboxes"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes max_num_gt_bboxes"]:
        """
        Compute pairwise classification cost between queries and labels. Since it's
        batch-wise implementation, the cost is computed for invalid gt_bboxes, but the cost will be
        highest for invalid gt_bboxes, so the assignment will not match to invalid gt_bboxes.dense_

        Args:
            cls_logits: Classification logits for each query.
            gt_labels: Ground-truth class labels.
            valid_masks: Mask indicating valid ground-truth labels.

        Returns:
            Pairwise classification cost matrix in:
            [
                [cost(pred_box_0, gt_box_0), cost(pred_box_0, gt_box_1), ...]
                [cost(pred_box_1, gt_box_0), cost(pred_box_1, gt_box_1), ...]
            ]
            for every batch.
        """
        num_queries = cls_logits.shape[1]
        probs = cls_logits.sigmoid().clamp(min=self.eps, max=1.0 - self.eps)
        neg_cost = -(1.0 - probs + self.eps).log() * (1.0 - self.alpha) * probs.pow(self.gamma)
        pos_cost = -(probs + self.eps).log() * self.alpha * (1.0 - probs).pow(self.gamma)

        # Pick each gt's own class channel out of the class axis. Padded gt_labels are -1
        # (see Detection3DGTBatch), which gather rejects, so they are masked out explicitly the
        # same way create_gaussian_heatmaps does. Any other label outside [0, num_classes) is a
        # real error, for example a taxonomy mismatch, and must not silently read another class.
        num_classes = cls_logits.shape[2]
        padded_labels = gt_labels == -1
        if bool((((gt_labels < 0) & ~padded_labels) | (gt_labels >= num_classes)).any()):
            raise ValueError(
                f"Found gt_labels outside [0, {num_classes - 1}] that are not the -1 padding. "
                "Every label must index a class channel of cls_logits."
            )
        # Padded labels index class 0 for a legal gather. Their columns are overwritten below,
        # so the gathered value itself is irrelevant.
        # (batch_size, num_gt_bboxes) -> (batch_size, num_queries, num_gt_bboxes)
        class_indices = (
            gt_labels.masked_fill(padded_labels, 0).unsqueeze(1).expand(-1, num_queries, -1)
        )
        cost = (pos_cost.gather(2, class_indices) - neg_cost.gather(2, class_indices)) * self.weight
        # Padded or invalid gts get the highest cost so the assignment never matches them.
        # (batch_size, num_gt_bboxes) -> (batch_size, 1, num_gt_bboxes), broadcast to every query
        cost = cost.masked_fill(
            (~valid_masks | padded_labels).unsqueeze(1), torch.finfo(cost.dtype).max
        )
        return cost


@dataclass(frozen=True)
class BBoxBEVL1Cost:
    """Compute normalized BEV L1 cost between proposal and target boxes.

    The cost operates on BEV box centers normalized by the detector range. It takes the whole
    boxes like :class:`BBox3DL1Cost`, so an assigner calls either cost the same way.
    """

    weight: float = 1.0

    def __call__(
        self,
        bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        gt_bboxes: Float32[torch.Tensor, "batch_size max_num_gt_bboxes code_size"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
        point_cloud_range: Sequence[float],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes max_num_gt_bboxes"]:
        """Compute pairwise BEV L1 cost in normalized coordinates.

        Like ClassificationCost, this is a batch-wise implementation: the cost is computed for
        invalid gt_bboxes too, but those columns are filled with the highest possible cost so the
        assignment never matches them.

        Args:
            bboxes: Predicted metric boxes, ``(x, y)`` centers first; only the BEV centers enter
                the cost.
            gt_bboxes: Ground-truth metric boxes in the same layout.
            valid_masks: Mask indicating valid ground-truth boxes.
            point_cloud_range: Detector point-cloud range.

        Returns:
            Pairwise BEV L1 cost matrix, one (num_predictions, num_gt_bboxes) block per
            batch element.
        """
        # (batch_size, num_predictions, 2) and (batch_size, num_gt_bboxes, 2)
        bboxes_centers = bboxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Y + 1]
        gt_bboxes_centers = gt_bboxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Y + 1]
        # (2,) BEV origin and extent used to normalize both center sets into [0, 1].
        pc_start = bboxes_centers.new_tensor(point_cloud_range[0:2])
        pc_extent = bboxes_centers.new_tensor(point_cloud_range[3:5]) - pc_start
        # Both center sets end in the (x, y) axis, so the normalization broadcasts over batch
        # and box axes.
        # (batch_size, num_predictions, 2) and (batch_size, num_gt_bboxes, 2)
        norm_bboxes = (bboxes_centers - pc_start) / pc_extent
        norm_gt_bboxes = (gt_bboxes_centers - pc_start) / pc_extent
        # (batch_size, num_predictions, 2) x (batch_size, num_gt_bboxes, 2) ->
        # (batch_size, num_predictions, num_gt_bboxes)
        pairwise_distances = torch.cdist(norm_bboxes, norm_gt_bboxes, p=1)
        pairwise_distances = pairwise_distances * self.weight
        # Invalid gt bboxes has highest distance (cost). The weight is applied first, the same way
        # ClassificationCost does, so the sentinel stays exactly finfo.max instead of being scaled
        # down by a small weight (or overflowing to inf for a weight above one).
        # valid_masks: (batch_size, num_gt_bboxes) -> (batch_size, 1, num_gt_bboxes)
        # Broadcast to every column (gt_bboxes)
        return pairwise_distances.masked_fill(
            ~valid_masks.unsqueeze(1),
            torch.finfo(pairwise_distances.dtype).max,
        )


@dataclass(frozen=True)
class BBox3DL1Cost:
    """Compute weighted L1 cost over full normalized 3D box encodings.

    Both proposals and ground truth are encoded with :func:`normalize_boxes3d`
    and every channel is scaled by ``code_weights`` before the pairwise L1
    distance. This is the DETR-style matching cost used by query-based camera
    detectors, where center, size, and heading all steer the assignment.
    """

    weight: float = 1.0
    code_weights: tuple[float, ...] = (1.0,) * 10

    def __call__(
        self,
        bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        gt_bboxes: Float32[torch.Tensor, "batch_size max_num_gt_bboxes code_size"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
        point_cloud_range: Sequence[float],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes max_num_gt_bboxes"]:
        """Compute pairwise weighted L1 cost in the normalized box encoding.

        Like the other costs, this is a batch-wise implementation: the columns of the padded
        gt boxes are filled with the highest possible cost so the assignment never matches them.

        Args:
            bboxes: Predicted metric boxes ``[cx, cy, cz, dx, dy, dz, yaw, vx, vy]``.
            gt_bboxes: Ground-truth metric boxes in the same layout, zeros where padded.
            valid_masks: Mask indicating valid ground-truth boxes.
            point_cloud_range: Unused; kept for the shared assigner interface.

        Returns:
            Pairwise weighted L1 cost matrix, one (num_bboxes, num_gt_bboxes) block per
            batch element.
        """
        del point_cloud_range
        channel_weights = bboxes.new_tensor(self.code_weights)
        # The encoding takes the log of the sizes, which is -inf for the all-zero padded boxes
        # and would spread as inf or nan over the whole distance row. The padded rows are
        # encoded as unit boxes instead; their columns are overwritten with the sentinel below.
        unit_box = gt_bboxes.new_zeros(gt_bboxes.shape[-1])
        unit_box[Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1] = 1.0
        safe_gt_bboxes = torch.where(valid_masks.unsqueeze(-1), gt_bboxes, unit_box)
        # (batch_size, num_bboxes, 10) and (batch_size, max_num_gt_bboxes, 10)
        norm_bboxes = normalize_boxes3d(bboxes) * channel_weights
        norm_gt_bboxes = normalize_boxes3d(safe_gt_bboxes) * channel_weights
        # (batch_size, num_bboxes, max_num_gt_bboxes); the weight is applied before the
        # sentinel so the sentinel stays exactly finfo.max.
        cost = torch.cdist(norm_bboxes, norm_gt_bboxes, p=1) * self.weight
        return cost.masked_fill(~valid_masks.unsqueeze(1), torch.finfo(cost.dtype).max)


@dataclass(frozen=True)
class IoU3DCost:
    """Compute a negative-IoU matching cost.

    Higher overlaps produce lower matching costs, making the term suitable for
    Hungarian assignment.
    """

    weight: float = 1.0

    def __call__(
        self, iou: Float32[torch.Tensor, "batch_size num_predictions max_num_gt_bboxes"]
    ) -> Float32[torch.Tensor, "batch_size num_predictions max_num_gt_bboxes"]:
        """Convert IoU values into a minimization cost.

        Args:
            iou: Pairwise IoU matrix.

        Returns:
            Pairwise minimization cost derived from IoU.
        """
        return -iou * self.weight
