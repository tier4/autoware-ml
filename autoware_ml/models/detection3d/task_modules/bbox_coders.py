"""Reusable box coders for detection3d heads.

This module implements reusable box encoding and decoding logic for 3D
detection heads and deployment paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

from jaxtyping import Bool, Float32, Int64
import torch


class ScoreThresholdConfig(NamedTuple):
    """
    Config for score threshold to each class group.
    Args:
        class_names: Names of classes for this group.
        score_threshold: Score thresold to keep bboxes in this group.
    """

    class_names: Sequence[str]
    score_threshold: float


class ScoreThresholdGroup(NamedTuple):
    """
    Resolved config for each score thresold group.
    Args:
        class_ids: Class indices for this group.
        score_threshold: Score threshold to keep bboxes in this group.
    """

    class_ids: Sequence[int]
    score_threshold: float


def normalize_boxes3d(boxes: Float32[torch.Tensor, " 9"]) -> torch.Tensor:
    """Encode metric 3D boxes into the query-regression space.

    Args:
        boxes: Metric boxes ``[cx, cy, cz, dx, dy, dz, yaw, vx, vy]``.

    Returns:
        Encoded boxes ``[cx, cy, cz, log dx, log dy, log dz, sin, cos, vx, vy]``.
    """
    return torch.cat(
        [
            boxes[..., :3],
            boxes[..., 3:6].log(),
            torch.sin(boxes[..., 6:7]),
            torch.cos(boxes[..., 6:7]),
            boxes[..., 7:9],
        ],
        dim=-1,
    )


def denormalize_boxes3d(boxes: Float32[torch.Tensor, " 10"]) -> torch.Tensor:
    """Decode query-regression boxes back into the metric space.

    Args:
        boxes: Encoded boxes ``[cx, cy, cz, log dx, log dy, log dz, sin, cos, vx, vy]``.

    Returns:
        Metric boxes ``[cx, cy, cz, dx, dy, dz, yaw, vx, vy]``.
    """
    return torch.cat(
        [
            boxes[..., :3],
            boxes[..., 3:6].exp(),
            torch.atan2(boxes[..., 6:7], boxes[..., 7:8]),
            boxes[..., 8:10],
        ],
        dim=-1,
    )


@dataclass
class TransFusionBBoxCoder:
    """Encode and decode boxes for TransFusion-style query heads.

    Attributes:
        pc_range: Point-cloud range used by the detector.
        out_size_factor: BEV downsampling factor between point space and feature space.
        voxel_size: Voxel size along each spatial axis.
        post_center_range: Optional metric-space range used to filter predictions.
        score_threshold: Optional score threshold applied during decoding. A
            scalar applies to every class; a sequence provides one threshold
            per class index.
        code_size: Number of regression channels produced by the head.
    """

    pc_range: list[float]
    out_size_factor: int
    voxel_size: list[float]
    score_threshold_groups: Sequence[ScoreThresholdGroup] | None
    post_center_range: Sequence[float] | None = None
    # score_threshold: float | Sequence[float] | None = None
    code_size: int = 8

    def encode(
        self,
        gt_boxes: Float32[torch.Tensor, "batch_size max_num_boxes 9"],
    ) -> Float32[torch.Tensor, "batch_size max_num_boxes code_size"]:
        """Encode metric-space boxes into normalized regression targets.

        Args:
            gt_boxes: Ground-truth boxes in metric coordinates, where each box is represented as
                (center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y).
                Note that it will also encode invalid boxes (e.g., zero-padded boxes) into
                regression targets, so the caller should mask out invalid boxes if necessary.

        Returns:
            Encoded regression targets aligned with the TransFusion head layout.
        """
        batch_size, max_num_boxes, _ = gt_boxes.shape
        targets = torch.zeros(
            (batch_size, max_num_boxes, self.code_size),
            device=gt_boxes.device,
            dtype=gt_boxes.dtype,
        )
        targets[:, :, 0] = (gt_boxes[:, :, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0]
        )
        targets[:, :, 1] = (gt_boxes[:, :, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1]
        )
        dims = gt_boxes[:, :, 3:6]
        log_dims = dims.log()
        targets[:, :, 3] = log_dims[:, :, 0]
        targets[:, :, 4] = log_dims[:, :, 1]
        targets[:, :, 5] = log_dims[:, :, 2]
        # NOTE: gt_bbox is always in gravity center, so it doesn't need to shift
        targets[:, :, 2] = gt_boxes[:, :, 2]
        targets[:, :, 6] = torch.sin(gt_boxes[:, :, 6])
        targets[:, :, 7] = torch.cos(gt_boxes[:, :, 6])
        if self.code_size == 10:
            targets[:, :, 8:10] = gt_boxes[:, :, 7:9]
        return targets

    def decode_heatmaps(
        self,
        heatmaps: Float32[torch.Tensor, "batch_size num_classes num_proposals"],
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_proposals"],
        Float32[torch.Tensor, "batch_size num_proposals"],
    ]:
        """Decode heatmaps into class predictions and scores.

        Args:
            heatmaps: Class confidence heatmap.
        """
        final_scores, final_preds = heatmaps.max(dim=1)
        return final_scores, final_preds

    def filter_bboxes_center_range(
        self,
        bbox_centers: Float32[torch.Tensor, "batch_size num_proposals 3"],
    ) -> Bool[torch.Tensor, "batch_size num_proposals"]:
        """
        Build a per-proposal keep mask from the detector's post-processing center range.

        The test is axis aligned rather than a distance: a proposal is kept when its center lies
        inside the (x_min, y_min, z_min, x_max, y_max, z_max) box, inclusive at both ends.

        Args:
            bbox_centers: Box centers as (center_x, center_y, center_z).

        Returns:
            True where the proposal's center is inside post_center_range, False otherwise. All
            True when no range is configured.
        """
        batch_size, num_bboxes, _ = bbox_centers.shape
        if self.post_center_range is None:
            return torch.ones(
                (batch_size, num_bboxes), device=bbox_centers.device, dtype=torch.bool
            )
        if len(self.post_center_range) != 6:
            raise ValueError(
                "TransFusionBBoxCoder post_center_range must hold 6 values "
                f"(x_min, y_min, z_min, x_max, y_max, z_max), got {len(self.post_center_range)}."
            )

        # (6,) -> (1, 1, 6) so the mins and maxs broadcast over batch and proposals.
        center_range = torch.tensor(
            self.post_center_range,
            dtype=bbox_centers.dtype,
            device=bbox_centers.device,
        ).view(1, 1, 6)

        # Reduce over the trailing coordinate axis: x, y and z must all be inside.
        # (batch_size, num_proposals, 3) -> (batch_size, num_proposals)
        keep_masks = (bbox_centers >= center_range[..., :3]).all(dim=-1)
        keep_masks &= (bbox_centers <= center_range[..., 3:]).all(dim=-1)
        return keep_masks

    def filter_bboxes_score(
        self,
        bbox_scores: Float32[torch.Tensor, "batch_size num_proposals"],
        bbox_classes: Int64[torch.Tensor, "batch_size num_proposals"],
        num_classes: int,
    ) -> Bool[torch.Tensor, "batch_size num_proposals"]:
        """
        Build a per-proposal keep mask by comparing each score against its class group's threshold.

        The groups carry class ids, so they are first flattened into a per-class threshold vector
        and then looked up by each proposal's predicted class. That keeps the comparison a single
        elementwise op over the whole batch, with no loop over samples or proposals, and exports as
        a constant Gather plus a Greater.

        Args:
            bbox_scores: Confidence score for each proposal.
            bbox_classes: Predicted class id for each proposal. Must be a valid class index in
                [0, num_classes), since it indexes the per-class threshold vector.
            num_classes: Total number of classes.

        Returns:
            True where the proposal's score clears its class threshold, False otherwise.
        """
        if self.score_threshold_groups is None:
            return torch.ones_like(bbox_scores, dtype=torch.bool)

        # (num_classes,) threshold per class. Classes no group claims stay at zero.
        class_thresholds = torch.zeros(
            num_classes, dtype=bbox_scores.dtype, device=bbox_scores.device
        )
        for group_index, group in enumerate(self.score_threshold_groups):
            for class_id in group.class_ids:
                class_id = int(class_id)
                if not 0 <= class_id < num_classes:
                    raise ValueError(
                        f"TransFusionBBoxCoder score threshold group {group_index} class id "
                        f"{class_id} is out of range for {num_classes} classes."
                    )
                class_thresholds[class_id] = group.score_threshold

        # (num_classes,)[(batch_size, num_proposals)] -> (batch_size, num_proposals)
        return bbox_scores > class_thresholds[bbox_classes]

    def decode_boxes(
        self,
        rots: Float32[torch.Tensor, "batch_size 2 num_proposals"],
        dims: Float32[torch.Tensor, "batch_size 3 num_proposals"],
        centers: Float32[torch.Tensor, "batch_size 2 num_proposals"],
        heights: Float32[torch.Tensor, "batch_size 1 num_proposals"],
        vels: Float32[torch.Tensor, "batch_size 2 num_proposals"] | None,
    ) -> Float32[torch.Tensor, "batch_size num_proposals code_size"]:
        """
        Decode regression channels into metric-space boxes, where each box is represented as
        (center_x, center_y, center_z, length, width, height, yaw,  velocity_x, velocity_y).

        Args:
            rots: Rotation channels storing sine and cosine values.
            dims: Log-space box dimensions.
            centers: Predicted BEV center offsets.
            heights: Predicted box gravity heights, where the box center
            is already at height + 0.5 * box_height.
            vels: Optional velocity channels.

        Returns:
            Decoded boxes in metric coordinates.
        """
        centers = centers.clone()
        dims = dims.clone()
        heights = heights.clone()
        centers[:, 0, :] = (
            centers[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        )
        centers[:, 1, :] = (
            centers[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        )
        dims = dims.exp()
        # NOTE: gt_bbox is always in gravity_center, so it doesn't need to shift
        # heights = heights - dims[:, 2:3, :] * 0.5
        yaw = torch.atan2(rots[:, 0:1, :], rots[:, 1:2, :])
        if vels is None:
            final_boxes = torch.cat([centers, heights, dims, yaw], dim=1)
        else:
            final_boxes = torch.cat([centers, heights, dims, yaw, vels], dim=1)
        return final_boxes.permute(0, 2, 1).contiguous()


@dataclass
class NMSFreeBBoxCoder3D:
    """Decode box predictions for query-based 3D detectors without NMS.

    Attributes:
        pc_range: Point-cloud range used by the detector.
        post_center_range: Optional metric-space range used to filter predictions.
        score_threshold: Optional confidence threshold applied during decoding.
        max_num: Maximum number of predictions retained per sample.
    """

    pc_range: list[float]
    post_center_range: list[float] | None = None
    score_threshold: float | None = None
    max_num: int = 100

    def decode(
        self,
        cls_logits: torch.Tensor,
        box_params: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode class logits and box parameters into metric-space predictions.

        Args:
            cls_logits: Classification logits for each query.
            box_params: Box regression outputs for each query.

        Returns:
            Per-sample prediction dictionaries with boxes, scores, and labels.
        """
        batch_predictions: list[dict[str, torch.Tensor]] = []
        center_range = None
        if self.post_center_range is not None:
            center_range = box_params.new_tensor(self.post_center_range)

        for sample_scores, sample_boxes in zip(cls_logits.sigmoid(), box_params):
            flat_scores = sample_scores.flatten()
            topk = min(self.max_num, flat_scores.numel())
            scores, flat_indices = flat_scores.topk(topk)
            labels = flat_indices % sample_scores.shape[1]
            box_indices = flat_indices // sample_scores.shape[1]
            selected_boxes = sample_boxes[box_indices]

            if self.score_threshold is not None:
                keep = scores >= self.score_threshold
                scores = scores[keep]
                labels = labels[keep]
                selected_boxes = selected_boxes[keep]

            metric_boxes = denormalize_boxes3d(selected_boxes)
            if center_range is not None and metric_boxes.numel() > 0:
                keep = (metric_boxes[:, :3] >= center_range[:3]).all(dim=1)
                keep &= (metric_boxes[:, :3] <= center_range[3:]).all(dim=1)
                metric_boxes = metric_boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            batch_predictions.append({"bboxes": metric_boxes, "scores": scores, "labels": labels})
        return batch_predictions
