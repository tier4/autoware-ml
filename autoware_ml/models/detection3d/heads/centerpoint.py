"""Detection heads used by CenterPoint-style models.

This module implements dense prediction heads, target generation, decoding,
and training losses used by CenterPoint-style detectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.common.layers.conv import ConvModule
from autoware_ml.models.detection3d.task_modules.heatmap import (
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
)


def _gather_feat(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather flattened features at the requested indices."""
    channels = features.shape[-1]
    expanded_indices = indices.unsqueeze(-1).expand(*indices.shape, channels)
    return features.gather(dim=1, index=expanded_indices)


def _transpose_and_gather_feat(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Transpose a feature map and gather flattened features."""
    features = features.permute(0, 2, 3, 1).contiguous()
    features = features.view(features.shape[0], -1, features.shape[-1])
    return _gather_feat(features, indices)


@dataclass
class CenterPointTargets:
    """Store dense heatmap and regression targets for CenterPoint.

    Attributes:
        heatmap: Dense class heatmap targets.
        anno_boxes: Encoded box regression targets.
        indices: Flattened feature-map indices of positive targets.
        mask: Mask indicating valid target slots.
    """

    heatmap: torch.Tensor
    anno_boxes: torch.Tensor
    indices: torch.Tensor
    mask: torch.Tensor


class CenterHead(nn.Module):
    """Predict dense heatmaps and regression maps for CenterPoint.

    The head uses a shared BEV tower followed by lightweight prediction
    branches for heatmap, center offsets, dimensions, rotation, and velocity.
    It also owns the CenterPoint target generation, loss computation, and
    decode logic so the model wrapper stays reusable and task-agnostic.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        shared_channels: int,
        point_cloud_range: list[float],
        voxel_size: list[float],
        out_size_factor: int,
        max_objs: int,
        min_radius: int,
        score_threshold: float,
        post_max_size: int,
        nms_min_radius: float,
        class_names: Sequence[str] | None = None,
        gaussian_overlap: float = 0.1,
        loss_bbox_weight: float = 0.25,
        heatmap_init_bias: float = -2.19,
        use_velocity: bool = True,
    ) -> None:
        """Initialize the CenterPoint head.

        Args:
            in_channels: Input feature channels.
            num_classes: Number of detection classes.
            shared_channels: Channel count for the shared tower.
            point_cloud_range: Detector point-cloud range.
            voxel_size: Voxel size used by preprocessing.
            out_size_factor: Downsampling factor between BEV cells and head outputs.
            max_objs: Maximum number of targets kept per sample.
            min_radius: Minimum Gaussian radius for heatmap targets.
            score_threshold: Score threshold applied during decoding.
            post_max_size: Maximum number of predictions kept after decoding.
            nms_min_radius: Minimum center distance used by circle NMS.
            class_names: Optional ordered class names for metric logging.
            gaussian_overlap: Minimum Gaussian overlap with the target box.
            loss_bbox_weight: Weight applied to the box regression loss.
            heatmap_init_bias: Initial bias used by the heatmap prediction branch.
            use_velocity: Whether to predict velocity components.
        """
        super().__init__()
        self.num_classes = num_classes
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.max_objs = max_objs
        self.min_radius = min_radius
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.nms_min_radius = nms_min_radius
        self.class_names = tuple(class_names) if class_names is not None else None
        self.gaussian_overlap = gaussian_overlap
        self.loss_bbox_weight = loss_bbox_weight
        self.heatmap_init_bias = heatmap_init_bias
        self.use_velocity = use_velocity
        self.box_code_size = 10 if use_velocity else 8

        self.shared_conv = ConvModule(in_channels, shared_channels)
        self.heatmap = self._build_head(shared_channels, num_classes, init_bias=heatmap_init_bias)
        self.reg = self._build_head(shared_channels, 2)
        self.height = self._build_head(shared_channels, 1)
        self.dim = self._build_head(shared_channels, 3)
        self.rot = self._build_head(shared_channels, 2)
        self.vel = self._build_head(shared_channels, 2) if use_velocity else None

        self.loss_heatmap = GaussianFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")

    def _build_head(
        self, in_channels: int, out_channels: int, init_bias: float | None = None
    ) -> nn.Sequential:
        """Build one CenterPoint prediction branch."""
        head = nn.Sequential(
            ConvModule(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        if init_bias is not None:
            nn.init.constant_(head[-1].bias, init_bias)
        return head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict dense heatmap and regression maps."""
        shared = self.shared_conv(x)
        outputs = {
            "heatmap": self.heatmap(shared),
            "reg": self.reg(shared),
            "height": self.height(shared),
            "dim": self.dim(shared),
            "rot": self.rot(shared),
        }
        if self.vel is not None:
            outputs["vel"] = self.vel(shared)
        return outputs

    def get_targets(
        self,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        feature_map_size: tuple[int, int],
        device: torch.device,
    ) -> CenterPointTargets:
        """Build heatmap and regression targets for one batch."""
        batch_size = len(gt_boxes)
        feature_height, feature_width = feature_map_size
        heatmap = torch.zeros(
            (batch_size, self.num_classes, feature_height, feature_width), device=device
        )
        anno_boxes = torch.zeros((batch_size, self.max_objs, self.box_code_size), device=device)
        indices = torch.zeros((batch_size, self.max_objs), device=device, dtype=torch.long)
        mask = torch.zeros((batch_size, self.max_objs), device=device, dtype=torch.bool)

        for batch_index, (sample_boxes, sample_labels) in enumerate(zip(gt_boxes, gt_labels)):
            sample_boxes = sample_boxes.to(device=device, dtype=torch.float32)
            sample_labels = sample_labels.to(device=device, dtype=torch.long)
            num_objects = min(sample_boxes.shape[0], self.max_objs)
            for object_index in range(num_objects):
                box = sample_boxes[object_index]
                label = int(sample_labels[object_index].item())
                center_x = (
                    (box[0] - self.point_cloud_range[0]) / self.voxel_size[0] / self.out_size_factor
                )
                center_y = (
                    (box[1] - self.point_cloud_range[1]) / self.voxel_size[1] / self.out_size_factor
                )
                if not (0 <= center_x < feature_width and 0 <= center_y < feature_height):
                    continue

                center = box.new_tensor([center_x, center_y])
                center_int = center.floor().to(torch.long)
                length_cells = box[3] / self.voxel_size[0] / self.out_size_factor
                width_cells = box[4] / self.voxel_size[1] / self.out_size_factor
                radius = max(
                    self.min_radius,
                    gaussian_radius(
                        (width_cells.item(), length_cells.item()), self.gaussian_overlap
                    ),
                )
                draw_heatmap_gaussian(
                    heatmap[batch_index, label],
                    (int(center_int[0].item()), int(center_int[1].item())),
                    radius,
                )

                indices[batch_index, object_index] = center_int[1] * feature_width + center_int[0]
                mask[batch_index, object_index] = True
                encoded_box = [
                    center[0] - center_int[0].float(),
                    center[1] - center_int[1].float(),
                    box[2],
                    box[3].log(),
                    box[4].log(),
                    box[5].log(),
                    torch.sin(box[6]),
                    torch.cos(box[6]),
                ]
                if self.use_velocity:
                    encoded_box.extend([box[7], box[8]])
                anno_boxes[batch_index, object_index] = torch.stack(encoded_box)

        return CenterPointTargets(
            heatmap=heatmap, anno_boxes=anno_boxes, indices=indices, mask=mask
        )

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute CenterPoint heatmap and box losses."""
        targets = self.get_targets(
            gt_boxes, gt_labels, outputs["heatmap"].shape[-2:], outputs["heatmap"].device
        )
        loss_heatmap = self.loss_heatmap(outputs["heatmap"], targets.heatmap)

        pred_parts = [outputs["reg"], outputs["height"], outputs["dim"], outputs["rot"]]
        if self.use_velocity:
            pred_parts.append(outputs["vel"])
        pred_boxes = torch.cat(pred_parts, dim=1)
        pred_boxes = _transpose_and_gather_feat(pred_boxes, targets.indices)
        bbox_mask = targets.mask.unsqueeze(-1).expand_as(targets.anno_boxes).float()
        loss_bbox = self.loss_bbox(pred_boxes, targets.anno_boxes) * bbox_mask
        loss_bbox = loss_bbox.sum() / bbox_mask.sum().clamp_min(1.0)
        total_loss = loss_heatmap + self.loss_bbox_weight * loss_bbox
        return {"loss": total_loss, "loss_heatmap": loss_heatmap, "loss_bbox": loss_bbox}

    def predict(self, outputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Decode dense head outputs into 3D boxes, scores, and labels."""
        heatmap = outputs["heatmap"].sigmoid()
        pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        heatmap = heatmap * (pooled == heatmap)

        batch_size, num_classes, _, width = heatmap.shape
        predictions = []
        for batch_index in range(batch_size):
            scores = heatmap[batch_index].reshape(num_classes, -1)
            topk = min(self.post_max_size, scores.shape[1])
            top_scores, top_indices = scores.topk(k=topk, dim=1)
            class_ids = (
                torch.arange(num_classes, device=heatmap.device).unsqueeze(1).expand_as(top_indices)
            )

            flat_scores = top_scores.reshape(-1)
            flat_indices = top_indices.reshape(-1)
            flat_classes = class_ids.reshape(-1)
            keep = flat_scores > self.score_threshold
            if keep.sum() == 0:
                predictions.append(
                    {
                        "bboxes_3d": heatmap.new_zeros((0, 9 if self.use_velocity else 7)),
                        "scores_3d": heatmap.new_zeros((0,)),
                        "labels_3d": heatmap.new_zeros((0,), dtype=torch.long),
                    }
                )
                continue

            flat_scores = flat_scores[keep]
            flat_indices = flat_indices[keep]
            flat_classes = flat_classes[keep]
            ys = torch.div(flat_indices, width, rounding_mode="floor")
            xs = flat_indices % width

            reg = outputs["reg"][batch_index].permute(1, 2, 0).reshape(-1, 2)[flat_indices]
            height_pred = (
                outputs["height"][batch_index].permute(1, 2, 0).reshape(-1, 1)[flat_indices]
            )
            dim = outputs["dim"][batch_index].permute(1, 2, 0).reshape(-1, 3)[flat_indices].exp()
            rot = outputs["rot"][batch_index].permute(1, 2, 0).reshape(-1, 2)[flat_indices]

            xs = (xs.to(reg.dtype) + reg[:, 0]) * self.out_size_factor * self.voxel_size[
                0
            ] + self.point_cloud_range[0]
            ys = (ys.to(reg.dtype) + reg[:, 1]) * self.out_size_factor * self.voxel_size[
                1
            ] + self.point_cloud_range[1]
            yaw = torch.atan2(rot[:, 0], rot[:, 1]).unsqueeze(1)

            box_parts = [xs.unsqueeze(1), ys.unsqueeze(1), height_pred, dim, yaw]
            if self.use_velocity:
                vel = outputs["vel"][batch_index].permute(1, 2, 0).reshape(-1, 2)[flat_indices]
                box_parts.append(vel)
            boxes = torch.cat(box_parts, dim=1)

            kept_indices = []
            for class_id in flat_classes.unique():
                class_mask = flat_classes == class_id
                class_keep = circle_nms(
                    boxes[class_mask],
                    flat_scores[class_mask],
                    min_radius=self.nms_min_radius,
                    post_max_size=self.post_max_size,
                )
                class_indices = class_mask.nonzero(as_tuple=False).squeeze(1)[class_keep]
                kept_indices.append(class_indices)

            kept_indices = (
                torch.cat(kept_indices, dim=0)
                if kept_indices
                else boxes.new_zeros((0,), dtype=torch.long)
            )
            if kept_indices.numel() > 0:
                ranking = flat_scores[kept_indices].argsort(descending=True)[: self.post_max_size]
                kept_indices = kept_indices[ranking]

            predictions.append(
                {
                    "bboxes_3d": boxes[kept_indices],
                    "scores_3d": flat_scores[kept_indices],
                    "labels_3d": flat_classes[kept_indices],
                }
            )
        return predictions

    def prepare_for_export(self) -> "CenterHead":
        """Return an export-ready copy of the head.

        Returns:
            Deep copy of the head in evaluation mode.
        """
        return deepcopy(self).eval()
