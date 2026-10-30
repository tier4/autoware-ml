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
from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Bool
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import FocalHead2DOutputs
from autoware_ml.losses.detection2d.losses import GIoULoss, QualityFocalLoss, WeightedL1Loss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
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
        class_names: Sequence[str],
        in_channels: int = 256,
        embed_dims: int = 256,
        stride: int = 16,
        assigner: HungarianAssigner2D | None = None,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 5.0,
        loss_iou_weight: float = 2.0,
        loss_centers2d_weight: float = 10.0,
        loss_centerness_weight: float = 1.0,
    ) -> None:
        """Initialize the auxiliary 2D head.

        Args:
            class_names: Ordered names of the detector classes. Their count is the number
                of classification channels.
            in_channels: Neck feature channels.
            embed_dims: Hidden channels of the shared conv towers.
            stride: Feature-map stride relative to the padded image.
            assigner: Hungarian assigner for proposal-target matching.
            loss_cls_weight: Quality-focal classification loss weight.
            loss_bbox_weight: Normalized box L1 loss weight.
            loss_iou_weight: GIoU loss weight.
            loss_centers2d_weight: Projected-center L1 loss weight.
            loss_centerness_weight: Gaussian-heatmap centerness loss weight.
        """
        super().__init__()
        if embed_dims % 32 != 0:
            raise ValueError(f"embed_dims must be a multiple of 32 (GroupNorm), got {embed_dims}.")
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.stride = stride
        self.assigner = assigner if assigner is not None else HungarianAssigner2D()

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
        self.cls = nn.Conv2d(embed_dims, self.num_classes, kernel_size=1)
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
        self.loss_centerness = GaussianFocalLoss()
        self.loss_centerness_weight = loss_centerness_weight

    def forward(
        self,
        img_features: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> FocalHead2DOutputs:
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

        return FocalHead2DOutputs(
            cls_scores=cls_logits,
            bbox_preds=pred_bboxes,
            centers_2d=pred_centers2d,
            centerness=centerness,
            image_height=int(image_height),
            image_width=int(image_width),
        )

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
    ) -> torch.Tensor:
        """Draw the centerness heatmap of one image on the CPU.

        Drawing reads every center back to the host, which on the GPU would synchronize the
        device once per object; one copy of the boxes and centers per image replaces that.
        """
        heatmap = torch.zeros(image_height // self.stride, image_width // self.stride)
        if gt_centers2d.numel() == 0:
            return heatmap
        boxes_and_centers = torch.cat([gt_bboxes, gt_centers2d], dim=1).detach().cpu()
        gt_bboxes, gt_centers2d = boxes_and_centers[:, :4], boxes_and_centers[:, 4:]
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
        for center, radius in zip((gt_centers2d / self.stride).tolist(), radii.tolist()):
            draw_heatmap_gaussian(heatmap, (int(center[0]), int(center[1])), int(radius))
        return heatmap

    def loss(
        self,
        outputs: FocalHead2DOutputs,
        gt_bboxes_2d: list[list[torch.Tensor]],
        gt_labels_2d: list[list[torch.Tensor]],
        centers_2d: list[list[torch.Tensor]],
        padding_mask: Bool[torch.Tensor, " batch_size"],
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the five auxiliary 2D losses.

        Args:
            outputs: Dense 2D predictions of the head.
            gt_bboxes_2d: Per-sample, per-camera 2D boxes, one unnormalized
                ``(x_min, y_min, x_max, y_max)`` row per box in pixels: the edge columns of
                the Box2DFieldIndex rows, without the center.
            gt_labels_2d: Per-sample, per-camera class labels.
            centers_2d: Per-sample, per-camera object centers, one ``(center_x, center_y)``
                row per box in pixels: the center columns of the Box2DFieldIndex rows.
            padding_mask: Flags the padding samples of a streaming batch (see
                ``ModelGTBatch.padding_mask``), which take no part in the loss. Their 2D
                boxes are left out, so they give no positive and no box, IoU or center
                term, and a padding sample counts as annotated with no class, so every
                token of its images gets a zero classification and centerness weight. The
                weights only scale the loss terms, so a batch of padding alone yields a zero
                loss over the whole graph.
            gt_annotated_class_masks: True where a class is annotated in a sample; every
                image of a sample weights its classification by the row of its sample, so
                the background terms of the unannotated classes are dropped, while a token
                matched to a box keeps every term. None supervises every class of every
                image.

        Returns:
            Loss dictionary with ``loss_*2d`` keys.

        Raises:
            ValueError: If the predictions and the ground truth cover different images, or
                the masks do not hold one entry per sample.
        """
        cls_scores = outputs.cls_scores
        bbox_preds = outputs.bbox_preds
        pred_centers2d = outputs.centers_2d
        centerness = outputs.centerness
        image_height, image_width = outputs.image_height, outputs.image_width
        device = cls_scores.device

        batch_size = len(gt_bboxes_2d)
        num_cams = len(gt_bboxes_2d[0])
        num_images = batch_size * num_cams
        if cls_scores.size(0) != num_images:
            raise ValueError(
                f"2D predictions cover {cls_scores.size(0)} images but annotations cover "
                f"{num_images}; check the 2D annotation transform."
            )

        def to_tensor(value: object, columns: int) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            return tensor.reshape(-1, columns)

        if gt_annotated_class_masks is not None and gt_annotated_class_masks.shape != (
            batch_size,
            self.num_classes,
        ):
            raise ValueError(
                "gt_annotated_class_masks must have shape (batch_size, num_classes) = "
                f"{(batch_size, self.num_classes)}, got {tuple(gt_annotated_class_masks.shape)}."
            )
        if padding_mask.shape != (batch_size,):
            raise ValueError(
                f"padding_mask must have shape (batch_size,) = ({batch_size},), got "
                f"{tuple(padding_mask.shape)}."
            )
        is_padding = padding_mask.tolist()
        real_samples = ~padding_mask.view(-1, 1)
        gt_annotated_class_masks = (
            real_samples.expand(-1, self.num_classes)
            if gt_annotated_class_masks is None
            else gt_annotated_class_masks & real_samples
        )
        # (batch, 1) -> (batch * num_cams, 1, 1): the images of a sample follow each other.
        image_weights = (
            real_samples.to(cls_scores.dtype).repeat_interleave(num_cams, dim=0).view(-1, 1, 1)
        )

        targets: list[_Targets2D] = []
        heatmaps = []
        for sample_index in range(batch_size):
            for camera_index in range(num_cams):
                image_index = sample_index * num_cams + camera_index
                gt_bboxes = to_tensor(gt_bboxes_2d[sample_index][camera_index], 4)
                gt_labels = torch.as_tensor(
                    gt_labels_2d[sample_index][camera_index], dtype=torch.long, device=device
                ).reshape(-1)
                gt_centers = to_tensor(centers_2d[sample_index][camera_index], 2)
                if is_padding[sample_index]:
                    # A padding sample only keeps the ranks in step; it has no ground truth.
                    gt_bboxes, gt_labels, gt_centers = gt_bboxes[:0], gt_labels[:0], gt_centers[:0]
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
                    self._build_heatmap(gt_centers, gt_bboxes, image_height, image_width)
                )

        labels = torch.cat([target.labels for target in targets], dim=0)
        bbox_targets = torch.cat([target.bbox_targets for target in targets], dim=0)
        bbox_weights = torch.cat([target.bbox_weights for target in targets], dim=0)
        centers2d_targets = torch.cat([target.centers2d_targets for target in targets], dim=0)
        # The positive count of the step averaged over the ranks and floored on its sum (see
        # reduce_mean_count), so below 1 when the step has fewer positives than ranks; the
        # losses divide by it as given. Every training rank runs this collective. The count is
        # float32 because under autocast the logits are bfloat16, which rounds counts above 256.
        local_pos = torch.tensor(float(sum(target.num_pos for target in targets)), device=device)
        num_total_pos = reduce_mean_count(local_pos).item()

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
        class_weights = self._class_weights(
            gt_annotated_class_masks, num_cams, cls_scores.size(1), cls_scores.dtype
        )
        if class_weights is not None:
            # A token matched to a box is vouched for by it and keeps every term, as in
            # TransFusion; only the background terms of the unannotated classes drop out.
            class_weights[labels < self.num_classes] = 1.0
        loss_cls = self.loss_cls2d(
            flat_cls_scores,
            labels,
            iou_scores,
            class_weights=class_weights,
            avg_factor=num_total_pos,
        )

        heatmap_targets = torch.stack(heatmaps, dim=0).to(device).view(num_images, -1, 1)
        # A padding image's empty heatmap would teach its tokens that no object is there.
        loss_centerness = self.loss_centerness_weight * self.loss_centerness(
            centerness, heatmap_targets, weights=image_weights, avg_factor=num_total_pos
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

    def _class_weights(
        self,
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None,
        num_cams: int,
        tokens_per_image: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Broadcast the annotated class mask of every sample to the tokens of its images."""
        if gt_annotated_class_masks is None:
            return None
        # (batch, C) -> (batch * num_cams, C): the images of a sample follow each other.
        per_image = gt_annotated_class_masks.repeat_interleave(num_cams, dim=0)
        return (
            per_image.unsqueeze(1)
            .expand(-1, tokens_per_image, -1)
            .reshape(-1, self.num_classes)
            .to(dtype=dtype)
        )
