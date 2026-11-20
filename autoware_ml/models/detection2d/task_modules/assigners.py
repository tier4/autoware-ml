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

"""Hungarian assigner of the dense 2D proposals of an image-plane head."""

from __future__ import annotations

from dataclasses import dataclass, field

from jaxtyping import Bool, Float32, Int64
import torch

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.models.detection2d.task_modules.match_costs import (
    BBoxL1Cost2D,
    Center2DL1Cost,
    IoUCost2D,
)
from autoware_ml.models.detection3d.task_modules.assigners import hungarian_match
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost


@dataclass(frozen=True)
class AssignResult2D:
    """Assignment of the dense proposals of a batch of images to their ground truth.

    Attributes:
        num_gts: Number of ground-truth boxes of every image.
        gt_inds: Assigned ground-truth index of every proposal, offset by one; background
            proposals carry 0.
        labels: Assigned class label of every proposal, -1 for background.
    """

    num_gts: Int64[torch.Tensor, " batch_size"]
    gt_inds: Int64[torch.Tensor, "batch_size num_queries"]
    labels: Int64[torch.Tensor, "batch_size num_queries"]


@dataclass(frozen=True)
class HungarianAssigner2D:
    """One-to-one assignment between dense 2D proposals and ground truth, image by image.

    The costs of every image of a batch are computed in one call, as in
    :class:`HungarianAssigner3D`; only the Hungarian solver runs per image. The default cost
    weights are the reference Focal-PETR recipe (2 / 5 / 2 / 10), the same values the auxiliary
    2D head uses as its loss weights.
    """

    cls_cost: ClassificationCost = field(default_factory=lambda: ClassificationCost(weight=2.0))
    reg_cost: BBoxL1Cost2D = field(default_factory=lambda: BBoxL1Cost2D(weight=5.0))
    iou_cost: IoUCost2D = field(default_factory=lambda: IoUCost2D(weight=2.0))
    centers2d_cost: Center2DL1Cost = field(default_factory=lambda: Center2DL1Cost(weight=10.0))

    def assign(
        self,
        bbox_pred: Float32[torch.Tensor, "batch_size num_queries 4"],
        cls_pred: Float32[torch.Tensor, "batch_size num_queries num_classes"],
        pred_centers2d: Float32[torch.Tensor, "batch_size num_queries 2"],
        gt_bboxes: Float32[torch.Tensor, "batch_size max_num_gts 4"],
        gt_labels: Int64[torch.Tensor, "batch_size max_num_gts"],
        gt_centers2d: Float32[torch.Tensor, "batch_size max_num_gts 2"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gts"],
        image_height: int,
        image_width: int,
    ) -> AssignResult2D:
        """Assign each proposal of every image to background (0) or a 1-based box index.

        Args:
            bbox_pred: Proposal boxes of every image as normalized ``(cx, cy, w, h)``.
            cls_pred: Proposal classification logits of every image.
            pred_centers2d: Normalized proposal centers of every image.
            gt_bboxes: Ground-truth boxes of every image in unnormalized ``(x1, y1, x2, y2)``
                pixels, padded to the same count.
            gt_labels: Ground-truth class indices of every image, -1 on the padding.
            gt_centers2d: Ground-truth projected centers of every image in unnormalized
                pixels, padded to the same count.
            valid_masks: True on the real boxes of every image, False on the padding.
            image_height: Image height in pixels.
            image_width: Image width in pixels.

        Returns:
            Assignment result over the proposals of every image.

        Raises:
            ValueError: If a box, logit or center tensor is not float32, or a real box carries
                a negative label or one not below the number of class channels.
        """
        # Match in float32 like the reference (force_fp32): a bfloat16 sigmoid saturates above
        # logit 6.2, and a bfloat16 box is off by up to 1/256 of the image.
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
        if bool(((gt_labels < 0) & valid_masks).any()):
            raise ValueError(
                "Found a negative gt_labels on a valid box; -1 marks the padding only."
            )

        batch_size, num_queries = bbox_pred.shape[:2]
        num_gts = valid_masks.sum(dim=1)
        assigned_gt_inds = bbox_pred.new_zeros((batch_size, num_queries), dtype=torch.long)
        assigned_labels = bbox_pred.new_full((batch_size, num_queries), -1, dtype=torch.long)
        if num_queries == 0 or gt_bboxes.shape[1] == 0:
            return AssignResult2D(num_gts=num_gts, gt_inds=assigned_gt_inds, labels=assigned_labels)

        factor = gt_bboxes.new_tensor([image_width, image_height, image_width, image_height])
        pred_xyxy = BaseBBoxes2D.from_cxcywh(bbox_pred.reshape(-1, 4)).bbox_params.view(
            bbox_pred.shape
        )
        cost = (
            self.cls_cost(cls_pred, gt_labels, valid_masks)
            + self.reg_cost(bbox_pred, gt_bboxes / factor)
            + self.iou_cost(pred_xyxy * factor, gt_bboxes)
            + self.centers2d_cost(pred_centers2d, gt_centers2d / factor[:2])
        )
        # The classification cost puts the float32 maximum on the padded columns, which the sum
        # turns infinite; the solver needs finite values.
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6).detach().cpu()
        host_valid_masks = valid_masks.cpu()
        host_gt_inds = torch.zeros((batch_size, num_queries), dtype=torch.long)
        for image_index in range(batch_size):
            if not bool(host_valid_masks[image_index].any()):
                continue
            matched_rows, matched_cols = hungarian_match(cost[image_index])
            # The solver fills min(num_queries, max_num_gts) pairs, padding columns included.
            valid_pairs = host_valid_masks[image_index, matched_cols]
            host_gt_inds[image_index, matched_rows[valid_pairs]] = matched_cols[valid_pairs] + 1
        assigned_gt_inds = host_gt_inds.to(bbox_pred.device)
        is_matched = assigned_gt_inds > 0
        matched_labels = gt_labels.gather(1, (assigned_gt_inds - 1).clamp(min=0))
        assigned_labels = torch.where(is_matched, matched_labels, assigned_labels)
        return AssignResult2D(num_gts=num_gts, gt_inds=assigned_gt_inds, labels=assigned_labels)
