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

"""Auxiliary 2D detection head of the camera 3D detectors.

The head predicts a class, the distances to the four box edges, a projected 3D center and a
center heatmap logit per class for every token of every camera's neck feature map. Its five
losses train the image features; inference never runs it. The ground truth is the projection of the 3D boxes
onto every camera (:class:`~autoware_ml.transforms.camera.annotations2d.LoadAnnotations2DFromBoxes3D`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Bool, Float, Float32, Int64
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import FocalHead2DOutputs
from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.losses.detection2d.quality_focal import QualityFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner2D
from autoware_ml.models.detection3d.task_modules.heatmap import draw_heatmap_gaussian
from autoware_ml.models.detection3d.task_modules.streaming import inverse_sigmoid
from autoware_ml.utils.distributed import reduce_mean_count


def _token_locations(
    feature_height: int,
    feature_width: int,
    stride: int,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> Float32[torch.Tensor, "feature_height feature_width 2"]:
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


def _decode_bbox_distances(
    locations: Float32[torch.Tensor, "1 feature_height feature_width 2"],
    bbox_distances: Float[torch.Tensor, "num_images feature_height feature_width 4"],
) -> Float32[torch.Tensor, "num_images feature_height feature_width 4"]:
    """Decode normalized token-to-edge distances into clamped ``(cx, cy, w, h)`` boxes.

    The distances are to the left, top, right and bottom edges of the box, in that order.
    """
    xyxy = torch.stack(
        [
            locations[..., 0] - bbox_distances[..., 0],
            locations[..., 1] - bbox_distances[..., 1],
            locations[..., 0] + bbox_distances[..., 2],
            locations[..., 1] + bbox_distances[..., 3],
        ],
        dim=-1,
    ).clamp(min=0.0, max=1.0)
    return BaseBBoxes2D(xyxy.reshape(-1, 4).float()).cxcywh.view(xyxy.shape)


def _decode_center_offset(
    locations: Float32[torch.Tensor, "1 feature_height feature_width 2"],
    center_offset: Float[torch.Tensor, "num_images feature_height feature_width 2"],
) -> Float32[torch.Tensor, "num_images feature_height feature_width 2"]:
    """Decode center offsets in inverse-sigmoid space into normalized centers."""
    return (inverse_sigmoid(locations) + center_offset).sigmoid()


def _check_image_ground_truth(
    gt_bboxes: Float32[torch.Tensor, "..."],
    gt_labels: Int64[torch.Tensor, "..."],
    gt_centers: Float32[torch.Tensor, "..."],
    sample_index: int,
    camera_index: int,
) -> None:
    """Check that the ground truth of one image is one row per box.

    Raises:
        ValueError: If the boxes are not ``(num_boxes, 4)``, the centers not ``(num_boxes, 2)``
            or the labels not ``(num_boxes,)``.
    """
    num_boxes = gt_bboxes.shape[0] if gt_bboxes.ndim == 2 else -1
    if (
        gt_bboxes.ndim != 2
        or gt_bboxes.shape[1] != 4
        or gt_centers.shape != (num_boxes, 2)
        or gt_labels.shape != (num_boxes,)
    ):
        raise ValueError(
            f"The 2D ground truth of sample {sample_index}, camera {camera_index} must be "
            "(num_boxes, 4) box edges, (num_boxes, 2) centers and (num_boxes,) labels, got "
            f"{tuple(gt_bboxes.shape)}, {tuple(gt_centers.shape)} and {tuple(gt_labels.shape)}; "
            "a Box2DFieldIndex row holds the edges and the center, split it first."
        )


@dataclass
class _Targets2D:
    """Assignment targets for one camera image."""

    labels: Int64[torch.Tensor, " num_tokens"]
    is_positive: Bool[torch.Tensor, " num_tokens"]
    bbox_targets: Float32[torch.Tensor, "num_tokens 4"]
    centers2d_targets: Float32[torch.Tensor, "num_tokens 2"]
    num_pos: int


class FocalHead2D(nn.Module):
    """Dense auxiliary 2D head over the multiview neck features.

    Two 3x3 conv + GroupNorm towers, one for the classification and one for the regression,
    each followed by 1x1 prediction convs: the class logits and the center heatmap logits on
    the first, the distances to the box edges and the projected 3D center on the second. The
    heatmap has one channel per class with a Gaussian around the projected 3D center of every
    box, so the classes a sample does not annotate can be left out of it like out of the
    classification.
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
        loss_heatmap_weight: float = 1.0,
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
            loss_heatmap_weight: Gaussian center heatmap loss weight.
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
        self.heatmap = nn.Conv2d(embed_dims, self.num_classes, kernel_size=1)
        self.bbox = nn.Conv2d(embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(embed_dims, 2, kernel_size=1)

        bias_init = -math.log((1.0 - 0.01) / 0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.heatmap.bias, bias_init)

        self.loss_cls = QualityFocalLoss(beta=2.0)
        self.loss_l1 = nn.L1Loss(reduction="none")
        self.loss_heatmap = GaussianFocalLoss()
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_iou_weight = loss_iou_weight
        self.loss_centers2d_weight = loss_centers2d_weight
        self.loss_heatmap_weight = loss_heatmap_weight

    def forward(
        self,
        img_features: Float[torch.Tensor, "batch_size num_cams channels height width"],
        image_height: int,
        image_width: int,
    ) -> FocalHead2DOutputs:
        """Predict dense 2D outputs for every camera.

        Args:
            img_features: Neck features ``(batch, num_cams, C, H, W)``.
            image_height: Padded image height in pixels.
            image_width: Padded image width in pixels.

        Returns:
            Dense per-token predictions flattened to ``(batch * num_cams, H * W, ...)``, with
            the token grid and the image size.

        Raises:
            ValueError: If a side of the feature map has neither ``image // stride`` nor
                ``ceil(image / stride)`` tokens, so the stride is not the neck's or the
                image sides are swapped.
        """
        batch_size, num_cams, _, feature_height, feature_width = img_features.shape
        for side, feature, image in (
            ("height", feature_height, image_height),
            ("width", feature_width, image_width),
        ):
            if feature not in (image // self.stride, math.ceil(image / self.stride)):
                raise ValueError(
                    f"The feature map has {feature} tokens along its {side} but the {image} px "
                    f"image {side} at stride {self.stride} gives {image // self.stride} or "
                    f"{math.ceil(image / self.stride)}; check the stride of the head against "
                    "the neck and the order of image_height and image_width."
                )
        x = img_features.flatten(0, 1)

        cls_feat = self.shared_cls(x)
        cls_logits = (
            self.cls(cls_feat)
            .permute(0, 2, 3, 1)
            .reshape(batch_size * num_cams, -1, self.num_classes)
        )
        heatmap_logits = (
            self.heatmap(cls_feat)
            .permute(0, 2, 3, 1)
            .reshape(batch_size * num_cams, -1, self.num_classes)
        )

        reg_feat = self.shared_reg(x)
        bbox_distances = self.bbox(reg_feat).permute(0, 2, 3, 1).contiguous().sigmoid()
        center_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()

        locations = _token_locations(
            feature_height, feature_width, self.stride, image_height, image_width, x.device
        ).unsqueeze(0)
        bbox_preds = _decode_bbox_distances(locations, bbox_distances).view(
            batch_size * num_cams, -1, 4
        )
        pred_centers2d = _decode_center_offset(locations, center_offset).view(
            batch_size * num_cams, -1, 2
        )

        return FocalHead2DOutputs(
            cls_scores=cls_logits,
            bbox_preds=bbox_preds,
            centers_2d=pred_centers2d,
            heatmap=heatmap_logits,
            feature_height=int(feature_height),
            feature_width=int(feature_width),
            image_height=int(image_height),
            image_width=int(image_width),
        )

    def _get_targets_single(
        self,
        cls_logits: Float32[torch.Tensor, "num_tokens num_classes"],
        bbox_pred: Float32[torch.Tensor, "num_tokens 4"],
        pred_centers2d: Float32[torch.Tensor, "num_tokens 2"],
        gt_bboxes: Float32[torch.Tensor, "num_gts 4"],
        gt_labels: Int64[torch.Tensor, " num_gts"],
        gt_centers2d: Float32[torch.Tensor, "num_gts 2"],
        image_height: int,
        image_width: int,
    ) -> _Targets2D:
        """Match the tokens of one image to its boxes and build the targets of every token.

        The predictions are a detached host copy: no gradient goes through the assignment.
        """
        num_tokens = bbox_pred.size(0)
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
        is_positive = assigned.gt_inds > 0
        pos_inds = torch.nonzero(is_positive, as_tuple=False).squeeze(-1)

        labels = gt_labels.new_full((num_tokens,), self.num_classes, dtype=torch.long)
        bbox_targets = bbox_pred.new_zeros((num_tokens, 4))
        centers2d_targets = bbox_pred.new_zeros((num_tokens, 2))
        if pos_inds.numel() > 0:
            matched_gt_inds = assigned.gt_inds[pos_inds] - 1
            labels[pos_inds] = gt_labels[matched_gt_inds].long()
            factor = bbox_pred.new_tensor([image_width, image_height, image_width, image_height])
            bbox_targets[pos_inds] = BaseBBoxes2D(gt_bboxes[matched_gt_inds] / factor).cxcywh
            centers2d_targets[pos_inds] = gt_centers2d[matched_gt_inds] / factor[:2]
        return _Targets2D(
            labels=labels,
            is_positive=is_positive,
            bbox_targets=bbox_targets,
            centers2d_targets=centers2d_targets,
            num_pos=int(pos_inds.numel()),
        )

    def _build_heatmap(
        self,
        gt_centers2d: Float32[torch.Tensor, "num_gts 2"],
        gt_bboxes: Float32[torch.Tensor, "num_gts 4"],
        gt_labels: Int64[torch.Tensor, " num_gts"],
        feature_height: int,
        feature_width: int,
    ) -> Float32[torch.Tensor, "num_classes feature_height feature_width"]:
        """Draw the center heatmap of one image from its host ground truth, one cell per token.

        Every box peaks in the channel of its class. A center on the right or bottom edge of
        the image, where the annotation transform clamps a center beyond it, peaks in the last
        cell, as a center on the left or top edge peaks in cell 0.
        """
        heatmap = torch.zeros(self.num_classes, feature_height, feature_width)
        if gt_centers2d.numel() == 0:
            return heatmap
        edge_distances = torch.cat(
            [
                gt_centers2d[:, 0:1] - gt_bboxes[:, 0:1],
                gt_centers2d[:, 1:2] - gt_bboxes[:, 1:2],
                gt_bboxes[:, 2:3] - gt_centers2d[:, 0:1],
                gt_bboxes[:, 3:4] - gt_centers2d[:, 1:2],
            ],
            dim=-1,
        )
        radii = torch.ceil(edge_distances.min(dim=-1).values / self.stride).clamp(min=1.0)
        for center, radius, label in zip(
            (gt_centers2d / self.stride).tolist(), radii.tolist(), gt_labels.tolist()
        ):
            cell = (
                min(int(center[0]), feature_width - 1),
                min(int(center[1]), feature_height - 1),
            )
            draw_heatmap_gaussian(heatmap[label], cell, int(radius))
        return heatmap

    def loss(
        self,
        outputs: FocalHead2DOutputs,
        gt_bboxes_2d: list[list[Float32[torch.Tensor, "num_gts 4"]]],
        gt_labels_2d: list[list[Int64[torch.Tensor, " num_gts"]]],
        gt_centers_2d: list[list[Float32[torch.Tensor, "num_gts 2"]]],
        padding_mask: Bool[torch.Tensor, " batch_size"],
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the five auxiliary 2D losses.

        Args:
            outputs: Dense 2D predictions of the head.
            gt_bboxes_2d: Per-sample, per-camera 2D boxes, one unnormalized
                ``(x_min, y_min, x_max, y_max)`` row per box in pixels: the edge columns of
                the Box2DFieldIndex rows, without the center.
            gt_labels_2d: Per-sample, per-camera class labels.
            gt_centers_2d: Per-sample, per-camera object centers, one ``(center_x, center_y)``
                row per box in pixels: the center columns of the Box2DFieldIndex rows.
            padding_mask: Flags the padding samples of a streaming batch (see
                ``ModelGTBatch.padding_mask``). Their boxes are left out and every token of
                their images gets a zero classification and heatmap weight, so they take no
                part in the loss; a batch of padding alone gives a zero loss with a graph.
            gt_annotated_class_masks: True where a class is annotated in a sample. The
                classification background terms and the heatmap channels of the unannotated
                classes are dropped on every token of the sample's images, so the head is not
                taught that those objects are absent; a token matched to a box keeps every
                classification term. None supervises every class.

        Returns:
            Loss dictionary with ``loss_*2d`` keys.

        Raises:
            ValueError: If the predictions and the ground truth cover different images, the
                three ground-truth lists do not hold the same samples and cameras, the masks do
                not hold one entry per sample, the ground truth of an image is not one row per
                box, or a box or center is not finite.
        """
        # The assigner and the losses take float32; under autocast the outputs are half.
        cls_scores = outputs.cls_scores.float()
        bbox_preds = outputs.bbox_preds.float()
        pred_centers2d = outputs.centers_2d.float()
        heatmap_logits = outputs.heatmap.float()
        image_height, image_width = outputs.image_height, outputs.image_width
        device = cls_scores.device

        batch_size = len(gt_bboxes_2d)
        num_cams = len(gt_bboxes_2d[0])
        num_images = batch_size * num_cams
        ground_truth_lists = (gt_bboxes_2d, gt_labels_2d, gt_centers_2d)
        if any(len(per_sample) != batch_size for per_sample in ground_truth_lists) or any(
            len(per_camera) != num_cams
            for per_sample in ground_truth_lists
            for per_camera in per_sample
        ):
            raise ValueError(
                "gt_bboxes_2d, gt_labels_2d and gt_centers_2d must hold the same samples and "
                f"cameras, {batch_size} samples of {num_cams} cameras as the first sample."
            )
        if cls_scores.size(0) != num_images:
            raise ValueError(
                f"2D predictions cover {cls_scores.size(0)} images but annotations cover "
                f"{num_images}; check the 2D annotation transform."
            )
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
        ground_truth: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for sample_index in range(batch_size):
            for camera_index in range(num_cams):
                gt_bboxes = gt_bboxes_2d[sample_index][camera_index].to(
                    device=device, dtype=torch.float32
                )
                gt_labels = gt_labels_2d[sample_index][camera_index].to(
                    device=device, dtype=torch.long
                )
                gt_centers = gt_centers_2d[sample_index][camera_index].to(
                    device=device, dtype=torch.float32
                )
                _check_image_ground_truth(
                    gt_bboxes, gt_labels, gt_centers, sample_index, camera_index
                )
                if is_padding[sample_index]:
                    gt_bboxes, gt_labels, gt_centers = gt_bboxes[:0], gt_labels[:0], gt_centers[:0]
                ground_truth.append((gt_bboxes, gt_labels, gt_centers))
        # The assignment and the heatmaps run on one host copy of the ground truth and of the
        # predictions: per image on the device, they would synchronize it several times each.
        counts = [boxes.shape[0] for boxes, _, _ in ground_truth]
        host_boxes_and_centers = torch.cat(
            [torch.cat([boxes, centers], dim=1) for boxes, _, centers in ground_truth], dim=0
        ).cpu()
        host_labels = torch.cat([labels for _, labels, _ in ground_truth], dim=0).cpu()
        # A non-finite target would fail far from the cause.
        if not bool(torch.isfinite(host_boxes_and_centers).all()):
            raise ValueError(
                "The 2D boxes or centers hold a non-finite value; the annotation transform must "
                "drop the boxes it cannot project."
            )
        host_cls_scores, host_bbox_preds, host_pred_centers2d = (
            torch.cat([cls_scores, bbox_preds, pred_centers2d], dim=-1)
            .detach()
            .cpu()
            .split([self.num_classes, 4, 2], dim=-1)
        )

        targets: list[_Targets2D] = []
        heatmaps = []
        for image_index, (boxes_and_centers, labels) in enumerate(
            zip(host_boxes_and_centers.split(counts), host_labels.split(counts))
        ):
            gt_bboxes, gt_centers = boxes_and_centers[:, :4], boxes_and_centers[:, 4:]
            targets.append(
                self._get_targets_single(
                    host_cls_scores[image_index],
                    host_bbox_preds[image_index],
                    host_pred_centers2d[image_index],
                    gt_bboxes,
                    labels,
                    gt_centers,
                    image_height,
                    image_width,
                )
            )
            heatmaps.append(
                self._build_heatmap(
                    gt_centers, gt_bboxes, labels, outputs.feature_height, outputs.feature_width
                )
            )

        labels = torch.cat([target.labels for target in targets], dim=0).to(device)
        is_positive = torch.cat([target.is_positive for target in targets], dim=0).to(device)
        positive_weights = is_positive.to(cls_scores.dtype)
        bbox_targets = torch.cat([target.bbox_targets for target in targets], dim=0).to(device)
        centers2d_targets = torch.cat([target.centers2d_targets for target in targets], dim=0).to(
            device
        )
        # The positive count of the step, averaged over the ranks and floored on its sum (see
        # reduce_mean_count); the losses divide by it as given. Every training rank runs the
        # collective; evaluation averages no gradient, so it uses the count of the rank.
        local_pos = torch.tensor(float(sum(target.num_pos for target in targets)), device=device)
        num_total_pos = (
            reduce_mean_count(local_pos) if self.training else local_pos.clamp(min=1.0)
        ).item()

        factor = cls_scores.new_tensor([image_width, image_height, image_width, image_height])
        flat_bbox_preds = bbox_preds.reshape(-1, 4)
        pred_xyxy = BaseBBoxes2D.from_cxcywh(flat_bbox_preds).bbox_params * factor
        target_xyxy = BaseBBoxes2D.from_cxcywh(bbox_targets).bbox_params * factor

        pred_boxes, target_boxes = BaseBBoxes2D(pred_xyxy), BaseBBoxes2D(target_xyxy)
        loss_iou = (
            self.loss_iou_weight
            * ((1.0 - pred_boxes.giou(target_boxes, is_aligned=True)) * positive_weights).sum()
            / num_total_pos
        )
        # The IoU is the classification target, not a gradient path for the box.
        iou_scores = target_boxes.iou(pred_boxes, is_aligned=True).detach()

        flat_cls_scores = cls_scores.reshape(-1, self.num_classes)
        class_weights = self._class_weights(
            gt_annotated_class_masks, num_cams, cls_scores.size(1), cls_scores.dtype
        )
        # A token matched to a box keeps every term.
        class_weights[is_positive] = 1.0
        loss_cls = self.loss_cls_weight * self.loss_cls(
            flat_cls_scores, labels, iou_scores, num_total_pos, weights=class_weights
        )

        # (num_images, num_classes, H, W) -> (num_images, H * W, num_classes), as the logits.
        heatmap_targets = torch.stack(heatmaps, dim=0).flatten(2).transpose(1, 2).to(device)
        # One weight per image and class: the channels a sample does not annotate drop out, as
        # do all channels of a padding sample.
        heatmap_weights = self._class_weights(
            gt_annotated_class_masks, num_cams, 1, cls_scores.dtype
        ).view(num_images, 1, self.num_classes)
        loss_heatmap = self.loss_heatmap_weight * self.loss_heatmap(
            heatmap_logits, heatmap_targets, weights=heatmap_weights, avg_factor=num_total_pos
        )

        loss_bbox = (
            self.loss_bbox_weight
            * (self.loss_l1(flat_bbox_preds, bbox_targets).sum(dim=-1) * positive_weights).sum()
            / num_total_pos
        )
        loss_centers2d = (
            self.loss_centers2d_weight
            * (
                self.loss_l1(pred_centers2d.reshape(-1, 2), centers2d_targets).sum(dim=-1)
                * positive_weights
            ).sum()
            / num_total_pos
        )

        return {
            "loss_cls2d": loss_cls,
            "loss_bbox2d": loss_bbox,
            "loss_iou2d": loss_iou,
            "loss_centers2d": loss_centers2d,
            "loss_heatmap2d": loss_heatmap,
        }

    def _class_weights(
        self,
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"],
        num_cams: int,
        tokens_per_image: int,
        dtype: torch.dtype,
    ) -> Float[torch.Tensor, "num_tokens num_classes"]:
        """Broadcast the annotated class mask of every sample to the tokens of its images."""
        # (batch, C) -> (batch * num_cams, C): the images of a sample follow each other.
        per_image = gt_annotated_class_masks.repeat_interleave(num_cams, dim=0)
        return (
            per_image.unsqueeze(1)
            .expand(-1, tokens_per_image, -1)
            .reshape(-1, self.num_classes)
            .to(dtype=dtype)
        )
