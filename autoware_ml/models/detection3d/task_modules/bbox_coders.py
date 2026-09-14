"""Reusable box coders for detection3d heads.

This module implements reusable box encoding and decoding logic for 3D
detection heads and deployment paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def normalize_boxes3d(boxes: torch.Tensor) -> torch.Tensor:
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


def denormalize_boxes3d(boxes: torch.Tensor) -> torch.Tensor:
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
    post_center_range: list[float] | None = None
    score_threshold: float | Sequence[float] | None = None
    code_size: int = 8

    def encode(self, dst_boxes: torch.Tensor) -> torch.Tensor:
        """Encode metric-space boxes into normalized regression targets.

        Args:
            dst_boxes: Ground-truth boxes in metric coordinates.

        Returns:
            Encoded regression targets aligned with the TransFusion head layout.
        """
        targets = torch.zeros(
            (dst_boxes.shape[0], self.code_size), device=dst_boxes.device, dtype=dst_boxes.dtype
        )
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0]
        )
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1]
        )
        dims = dst_boxes[:, 3:6]
        log_dims = dims.log()
        targets[:, 3] = log_dims[:, 0]
        targets[:, 4] = log_dims[:, 1]
        targets[:, 5] = log_dims[:, 2]
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:9]
        return targets

    def decode(
        self,
        heatmap: torch.Tensor,
        rot: torch.Tensor,
        dim: torch.Tensor,
        center: torch.Tensor,
        height: torch.Tensor,
        vel: torch.Tensor | None,
        filter_predictions: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode head outputs into metric-space 3D boxes.

        Args:
            heatmap: Class confidence heatmap.
            rot: Rotation channels storing sine and cosine values.
            dim: Log-space box dimensions.
            center: Predicted BEV center offsets.
            height: Predicted box bottom heights.
            vel: Optional velocity channels.
            filter_predictions: Whether to apply score and range filtering.

        Returns:
            Per-sample prediction dictionaries with boxes, scores, and labels.
        """
        final_scores, final_preds = heatmap.max(dim=1)

        center = center.clone()
        dim = dim.clone()
        height = height.clone()
        center[:, 0, :] = (
            center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        )
        center[:, 1, :] = (
            center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        )
        dim = dim.exp()
        height = height - dim[:, 2:3, :] * 0.5
        yaw = torch.atan2(rot[:, 0:1, :], rot[:, 1:2, :])

        if vel is None:
            final_boxes = torch.cat([center, height, dim, yaw], dim=1).permute(0, 2, 1)
        else:
            final_boxes = torch.cat([center, height, dim, yaw, vel], dim=1).permute(0, 2, 1)

        predictions = []
        if self.score_threshold is None:
            threshold_mask = None
        elif isinstance(self.score_threshold, (int, float)):
            threshold_mask = final_scores > self.score_threshold
        else:
            thresholds = torch.tensor(
                list(self.score_threshold), device=heatmap.device, dtype=final_scores.dtype
            )
            if thresholds.shape[0] != heatmap.shape[1]:
                raise ValueError(
                    "Per-class score_threshold must provide one value per class: "
                    f"got {thresholds.shape[0]} thresholds for {heatmap.shape[1]} classes."
                )
            threshold_mask = final_scores > thresholds[final_preds]
        center_range = None
        if self.post_center_range is not None:
            center_range = torch.tensor(
                self.post_center_range, device=heatmap.device, dtype=final_boxes.dtype
            )

        for batch_index in range(heatmap.shape[0]):
            boxes = final_boxes[batch_index]
            scores = final_scores[batch_index]
            labels = final_preds[batch_index]
            if filter_predictions:
                mask = torch.ones(scores.shape[0], device=scores.device, dtype=torch.bool)
                if threshold_mask is not None:
                    mask &= threshold_mask[batch_index]
                if center_range is not None:
                    mask &= (boxes[:, :3] >= center_range[:3]).all(dim=1)
                    mask &= (boxes[:, :3] <= center_range[3:]).all(dim=1)
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
            predictions.append({"bboxes": boxes, "scores": scores, "labels": labels})
        return predictions


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
