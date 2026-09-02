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

"""Auxiliary 2D detection head for camera 3D detectors (Focal-PETR-style).

The head predicts per-token class scores, centerness, LTRB boxes, and
projected 3D centers on every camera's neck feature map. Its five losses
shape the image features during training; inference never runs it, so it adds
no deployment cost. The reference FocalHead's top-k token pruning
(``sample_weight`` / ``topk_indexes``) is intentionally not ported: the 3D
head consumes every token, this head is loss-only. Ground truth comes from
projecting the 3D boxes onto each camera
(:class:`autoware_ml.transforms.camera.annotations2d.LoadAnnotations2DFromBoxes3D`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.losses.detection2d.losses import (
    GIoULoss,
    HeatmapGaussianFocalLoss,
    QualityFocalLoss,
    WeightedL1Loss,
)
from autoware_ml.models.detection3d.partial_ignore import (
    mask_ignored_columns,
    normalize_status_flags,
    resolve_partial_ignore_labels,
)
from autoware_ml.models.detection3d.task_modules.assigners2d import HungarianAssigner2D
from autoware_ml.models.detection3d.task_modules.boxes2d import (
    bbox_cxcywh_to_xyxy,
    bbox_overlaps,
    bbox_xyxy_to_cxcywh,
)
from autoware_ml.models.detection3d.task_modules.heatmap import draw_heatmap_gaussian
from autoware_ml.models.detection3d.task_modules.streaming import (
    inverse_sigmoid,
    reduce_mean_count,
)


def _token_locations(
    feature_height: int,
    feature_width: int,
    stride: int,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Build normalized pixel-center locations for every feature-map token.

    Returns:
        Locations with shape ``(feature_height, feature_width, 2)`` as
        ``(x, y)`` normalized by the padded image size.
    """
    shifts_x = (
        torch.arange(0, stride * feature_width, step=stride, dtype=torch.float32, device=device)
        + stride // 2
    ) / image_width
    shifts_y = (
        torch.arange(0, stride * feature_height, step=stride, dtype=torch.float32, device=device)
        + stride // 2
    ) / image_height
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack((shift_x, shift_y), dim=-1)


def _apply_ltrb(locations: torch.Tensor, pred_ltrb: torch.Tensor) -> torch.Tensor:
    """Decode normalized LTRB distances into clamped ``(cx, cy, w, h)`` boxes."""
    pred_boxes = torch.stack(
        [
            locations[..., 0] - pred_ltrb[..., 0],
            locations[..., 1] - pred_ltrb[..., 1],
            locations[..., 0] + pred_ltrb[..., 2],
            locations[..., 1] + pred_ltrb[..., 3],
        ],
        dim=-1,
    ).clamp(min=0.0, max=1.0)
    return bbox_xyxy_to_cxcywh(pred_boxes)


def _apply_center_offset(locations: torch.Tensor, center_offset: torch.Tensor) -> torch.Tensor:
    """Decode center offsets in inverse-sigmoid space into normalized centers."""
    return (inverse_sigmoid(locations) + center_offset).sigmoid()


@dataclass
class _Targets2D:
    """Assignment targets for one camera image."""

    labels: torch.Tensor
    bbox_targets: torch.Tensor
    bbox_weights: torch.Tensor
    centers2d_targets: torch.Tensor
    num_pos: int


class FocalHead2D(nn.Module):
    """Dense auxiliary 2D head over multiview neck features.

    Ported from the StreamPETR ``FocalHead`` reference: shared 3x3 conv +
    GroupNorm towers for classification and regression, with 1x1 prediction
    convs for class logits, centerness, LTRB boxes, and projected 3D centers.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        embed_dims: int = 256,
        stride: int = 16,
        assigner: HungarianAssigner2D | None = None,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 5.0,
        loss_iou_weight: float = 2.0,
        loss_centers2d_weight: float = 10.0,
        loss_centerness_weight: float = 1.0,
        class_names: list[str] | None = None,
        partial_ignore_classes: list[str] | None = None,
    ) -> None:
        """Initialize the auxiliary 2D head.

        Args:
            num_classes: Number of detector classes.
            in_channels: Neck feature channels.
            embed_dims: Hidden channels of the shared conv towers.
            stride: Feature-map stride relative to the padded image.
            assigner: Hungarian assigner for proposal-target matching.
            loss_cls_weight: Quality-focal classification loss weight.
            loss_bbox_weight: Normalized box L1 loss weight.
            loss_iou_weight: GIoU loss weight.
            loss_centers2d_weight: Projected-center L1 loss weight.
            loss_centerness_weight: Gaussian-heatmap centerness loss weight.
            class_names: Ordered detector class names (for partial-ignore).
            partial_ignore_classes: Class names that are only partially
                annotated across scenes.
        """
        super().__init__()
        if embed_dims % 32 != 0:
            raise ValueError(f"embed_dims must be a multiple of 32 (GroupNorm), got {embed_dims}.")
        self.num_classes = num_classes
        self.stride = stride
        self.assigner = assigner if assigner is not None else HungarianAssigner2D()
        self.partial_ignore_labels = resolve_partial_ignore_labels(
            class_names, partial_ignore_classes
        )

        self.shared_cls = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=embed_dims),
            nn.ReLU(),
        )
        self.shared_reg = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=embed_dims),
            nn.ReLU(),
        )
        self.cls = nn.Conv2d(embed_dims, num_classes, kernel_size=1)
        self.centerness = nn.Conv2d(embed_dims, 1, kernel_size=1)
        self.ltrb = nn.Conv2d(embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(embed_dims, 2, kernel_size=1)

        bias_init = -math.log((1.0 - 0.01) / 0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

        self.loss_cls2d = QualityFocalLoss(beta=2.0, loss_weight=loss_cls_weight)
        self.loss_bbox2d = WeightedL1Loss(loss_weight=loss_bbox_weight)
        self.loss_iou2d = GIoULoss(loss_weight=loss_iou_weight)
        self.loss_centers2d = WeightedL1Loss(loss_weight=loss_centers2d_weight)
        self.loss_centerness = HeatmapGaussianFocalLoss(loss_weight=loss_centerness_weight)

    def forward(
        self,
        img_features: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> dict[str, Any]:
        """Predict dense 2D outputs for every camera.

        Args:
            img_features: Neck features ``(batch, num_cams, C, H, W)``.
            image_height: Padded image height in pixels.
            image_width: Padded image width in pixels.

        Returns:
            Dense per-token predictions flattened to
            ``(batch * num_cams, H * W, ...)``.
        """
        batch_size, num_cams, _, feature_height, feature_width = img_features.shape
        x = img_features.flatten(0, 1)

        cls_feat = self.shared_cls(x)
        cls_logits = (
            self.cls(cls_feat)
            .permute(0, 2, 3, 1)
            .reshape(batch_size * num_cams, -1, self.num_classes)
        )
        centerness = (
            self.centerness(cls_feat).permute(0, 2, 3, 1).reshape(batch_size * num_cams, -1, 1)
        )

        reg_feat = self.shared_reg(x)
        ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous().sigmoid()
        centers2d_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()

        locations = _token_locations(
            feature_height, feature_width, self.stride, image_height, image_width, x.device
        )[None]
        pred_bboxes = _apply_ltrb(locations, ltrb).view(batch_size * num_cams, -1, 4)
        pred_centers2d = _apply_center_offset(locations, centers2d_offset).view(
            batch_size * num_cams, -1, 2
        )

        return {
            "enc_cls_scores": cls_logits,
            "enc_bbox_preds": pred_bboxes,
            "pred_centers2d": pred_centers2d,
            "centerness": centerness,
            "pad_shape_2d": (image_height, image_width),
        }

    def _get_targets_single(
        self,
        cls_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        pred_centers2d: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_centers2d: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> _Targets2D:
        num_queries = bbox_pred.size(0)
        assigned = self.assigner.assign(
            bbox_pred,
            cls_logits,
            pred_centers2d,
            gt_bboxes,
            gt_labels,
            gt_centers2d,
            image_height,
            image_width,
        )
        pos_inds = torch.nonzero(assigned.gt_inds > 0, as_tuple=False).squeeze(-1)

        labels = gt_labels.new_full((num_queries,), self.num_classes, dtype=torch.long)
        bbox_targets = bbox_pred.new_zeros((num_queries, 4))
        bbox_weights = bbox_pred.new_zeros((num_queries, 4))
        centers2d_targets = bbox_pred.new_zeros((num_queries, 2))
        if pos_inds.numel() > 0:
            matched_gt_inds = assigned.gt_inds[pos_inds] - 1
            labels[pos_inds] = gt_labels[matched_gt_inds].long()
            factor = bbox_pred.new_tensor([image_width, image_height, image_width, image_height])
            bbox_targets[pos_inds] = bbox_xyxy_to_cxcywh(gt_bboxes[matched_gt_inds] / factor)
            bbox_weights[pos_inds] = 1.0
            centers2d_targets[pos_inds] = gt_centers2d[matched_gt_inds] / factor[:2]
        return _Targets2D(
            labels=labels,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            centers2d_targets=centers2d_targets,
            num_pos=int(pos_inds.numel()),
        )

    def _build_heatmap(
        self,
        gt_centers2d: torch.Tensor,
        gt_bboxes: torch.Tensor,
        image_height: int,
        image_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        heatmap = torch.zeros(
            image_height // self.stride, image_width // self.stride, device=device
        )
        if gt_centers2d.numel() == 0:
            return heatmap
        bounds = torch.cat(
            [
                gt_centers2d[:, 0:1] - gt_bboxes[:, 0:1],
                gt_centers2d[:, 1:2] - gt_bboxes[:, 1:2],
                gt_bboxes[:, 2:3] - gt_centers2d[:, 0:1],
                gt_bboxes[:, 3:4] - gt_centers2d[:, 1:2],
            ],
            dim=-1,
        )
        radii = torch.ceil(bounds.min(dim=-1).values / self.stride).clamp(min=1.0)
        for center, radius in zip(gt_centers2d / self.stride, radii.tolist()):
            draw_heatmap_gaussian(
                heatmap, (int(center[0].item()), int(center[1].item())), int(radius)
            )
        return heatmap

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_bboxes_2d: list[list[torch.Tensor]],
        gt_labels_2d: list[list[torch.Tensor]],
        centers_2d: list[list[torch.Tensor]],
        annotation_status: list[bool] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the five auxiliary 2D losses.

        Args:
            outputs: Model outputs holding the dense 2D predictions.
            gt_bboxes_2d: Per-sample, per-camera projected 2D boxes in
                unnormalized ``(x1, y1, x2, y2)`` pixels.
            gt_labels_2d: Per-sample, per-camera class labels.
            centers_2d: Per-sample, per-camera projected 3D centers in pixels.
            annotation_status: Per-sample annotation-completeness flags for
                partial-ignore; required when ``partial_ignore_classes`` is
                configured.

        Returns:
            Loss dictionary with ``loss_*2d`` keys.
        """
        cls_scores = outputs["enc_cls_scores"]
        bbox_preds = outputs["enc_bbox_preds"]
        pred_centers2d = outputs["pred_centers2d"]
        centerness = outputs["centerness"]
        image_height, image_width = outputs["pad_shape_2d"]
        device = cls_scores.device

        batch_size = len(gt_bboxes_2d)
        num_cams = len(gt_bboxes_2d[0])
        num_images = batch_size * num_cams
        if cls_scores.size(0) != num_images:
            raise ValueError(
                f"2D predictions cover {cls_scores.size(0)} images but annotations cover "
                f"{num_images}; check the 2D annotation transform and collation."
            )

        def to_tensor(value: object, columns: int) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            return tensor.reshape(-1, columns)

        targets: list[_Targets2D] = []
        heatmaps = []
        image_status: list[bool] = []
        if self.partial_ignore_labels is not None:
            annotation_status = normalize_status_flags(annotation_status, batch_size)
        else:
            annotation_status = [True] * batch_size
        for sample_index in range(batch_size):
            for camera_index in range(num_cams):
                image_index = sample_index * num_cams + camera_index
                gt_bboxes = to_tensor(gt_bboxes_2d[sample_index][camera_index], 4)
                gt_labels = torch.as_tensor(
                    gt_labels_2d[sample_index][camera_index], dtype=torch.long, device=device
                ).reshape(-1)
                gt_centers = to_tensor(centers_2d[sample_index][camera_index], 2)
                targets.append(
                    self._get_targets_single(
                        cls_scores[image_index],
                        bbox_preds[image_index],
                        pred_centers2d[image_index],
                        gt_bboxes,
                        gt_labels,
                        gt_centers,
                        image_height,
                        image_width,
                    )
                )
                heatmaps.append(
                    self._build_heatmap(gt_centers, gt_bboxes, image_height, image_width, device)
                )
                image_status.append(annotation_status[sample_index])

        labels = torch.cat([target.labels for target in targets], dim=0)
        bbox_targets = torch.cat([target.bbox_targets for target in targets], dim=0)
        bbox_weights = torch.cat([target.bbox_weights for target in targets], dim=0)
        centers2d_targets = torch.cat([target.centers2d_targets for target in targets], dim=0)
        # Global (cross-rank) positive count; see reduce_mean_count. Safe
        # collective: this method runs unconditionally on every training rank.
        local_pos = cls_scores.new_tensor(float(sum(target.num_pos for target in targets)))
        num_total_pos = torch.clamp(reduce_mean_count(local_pos), min=1.0).item()

        factor = cls_scores.new_tensor([image_width, image_height, image_width, image_height])
        flat_bbox_preds = bbox_preds.reshape(-1, 4)
        pred_xyxy = bbox_cxcywh_to_xyxy(flat_bbox_preds) * factor
        target_xyxy = bbox_cxcywh_to_xyxy(bbox_targets) * factor

        loss_iou = self.loss_iou2d(
            pred_xyxy.float(), target_xyxy.float(), bbox_weights.float(), avg_factor=num_total_pos
        )
        iou_scores = bbox_overlaps(
            target_xyxy.float(), pred_xyxy.float(), mode="iou", is_aligned=True
        ).detach()

        flat_cls_scores = cls_scores.reshape(-1, self.num_classes)
        class_weights = self._partial_ignore_class_weights(
            flat_cls_scores, image_status, cls_scores.size(1)
        )
        loss_cls = self.loss_cls2d(
            flat_cls_scores,
            labels,
            iou_scores,
            class_weights=class_weights,
            avg_factor=num_total_pos,
        )

        heatmap_targets = torch.stack(heatmaps, dim=0).view(num_images, -1, 1)
        centerness_prob = centerness.sigmoid().clamp(min=1e-4, max=1 - 1e-4)
        loss_centerness = self.loss_centerness(
            centerness_prob, heatmap_targets, avg_factor=num_total_pos
        )

        loss_bbox = self.loss_bbox2d(
            flat_bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
        loss_centers2d = self.loss_centers2d(
            pred_centers2d.reshape(-1, 2),
            centers2d_targets,
            bbox_weights[:, 0:2],
            avg_factor=num_total_pos,
        )

        return {
            "loss_cls2d": loss_cls,
            "loss_bbox2d": loss_bbox,
            "loss_iou2d": loss_iou,
            "loss_centers2d": loss_centers2d,
            "loss_centerness2d": loss_centerness,
        }

    def _partial_ignore_class_weights(
        self,
        flat_cls_scores: torch.Tensor,
        image_status: list[bool],
        tokens_per_image: int,
    ) -> torch.Tensor | None:
        """Zero partially annotated class columns on unannotated images."""
        if self.partial_ignore_labels is None or all(image_status):
            return None
        class_weights = flat_cls_scores.new_ones(flat_cls_scores.shape)
        for image_index, status in enumerate(image_status):
            if not status:
                start = image_index * tokens_per_image
                rows = torch.arange(start, start + tokens_per_image, device=flat_cls_scores.device)
                mask_ignored_columns(class_weights, rows, self.partial_ignore_labels)
        return class_weights
