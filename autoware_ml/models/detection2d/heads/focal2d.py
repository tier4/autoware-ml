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
losses train the image features; inference never runs it. The ground truth is the projection of
the 3D boxes onto every camera
(:class:`~autoware_ml.transforms.camera.annotations2d.LoadAnnotations2DFromBoxes3D`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Bool, Float, Float32, Int64
import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch
from autoware_ml.dataclasses.models.detection2d.head_outputs import FocalHead2DOutputs
from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.losses.detection2d.quality_focal import QualityFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.detection2d.task_modules.assigners import HungarianAssigner2D
from autoware_ml.models.detection3d.task_modules.heatmap import create_gaussian_heatmaps
from autoware_ml.models.detection3d.task_modules.streaming import inverse_sigmoid
from autoware_ml.types.geometry import Box2DFieldIndex
from autoware_ml.utils.distributed import reduce_mean_count


@dataclass
class _GroundTruth2D:
    """The 2D ground truth of every image of a batch, padded to the same number of boxes.

    Attributes:
        bboxes: ``(x_min, y_min, x_max, y_max)`` of every box in pixels.
        labels: Class index of every box, -1 on the padding.
        centers: Projected 3D center of every box in pixels.
        valid_masks: True on the real boxes, False on the padding and on every box of a
            padding sample.
    """

    bboxes: Float32[torch.Tensor, "num_images max_num_boxes 4"]
    labels: Int64[torch.Tensor, "num_images max_num_boxes"]
    centers: Float32[torch.Tensor, "num_images max_num_boxes 2"]
    valid_masks: Bool[torch.Tensor, "num_images max_num_boxes"]


@dataclass
class _Targets2D:
    """Assignment targets of every token of a batch, concatenated over its images.

    Attributes:
        labels: Class index of the box matched to every token, ``num_classes`` for none.
        is_positive: True on the tokens matched to a box.
        bbox_targets: ``(cx, cy, w, h)`` of the matched box normalized by the image size.
        centers2d_targets: Projected center of the matched box normalized by the image size.
        num_pos: Number of matched tokens.
    """

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
        assigner: HungarianAssigner2D = HungarianAssigner2D(),
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
            stride: Feature-map stride relative to the image, which must be a multiple of it.
            assigner: Hungarian assigner for proposal-target matching; the default matches
                with the reference cost weights.
            loss_cls_weight: Quality-focal classification loss weight.
            loss_bbox_weight: Normalized box L1 loss weight.
            loss_iou_weight: GIoU loss weight.
            loss_centers2d_weight: Projected-center L1 loss weight.
            loss_heatmap_weight: Gaussian center heatmap loss weight.

        Raises:
            ValueError: If ``embed_dims`` is not a multiple of the 32 GroupNorm groups.
        """
        super().__init__()
        if embed_dims % 32 != 0:
            raise ValueError(f"embed_dims must be a multiple of 32 (GroupNorm), got {embed_dims}.")
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.stride = stride
        self.assigner = assigner

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
            image_height: Image height in pixels.
            image_width: Image width in pixels.

        Returns:
            Dense float32 predictions per token flattened to ``(batch * num_cams, H * W, ...)``,
            with the token grid and the image size.

        Raises:
            ValueError: If the token grid times the stride is not the image size, so the stride
                is not the neck's, the image sides are swapped or the image is not a multiple of
                the stride.
        """
        batch_size, num_cams, _, feature_height, feature_width = img_features.shape
        if (feature_height * self.stride, feature_width * self.stride) != (
            image_height,
            image_width,
        ):
            raise ValueError(
                f"The {feature_height} x {feature_width} token grid at stride {self.stride} "
                f"covers {feature_height * self.stride} x {feature_width * self.stride} px but "
                f"the image is {image_height} x {image_width}; the stride must be the neck's "
                "and the image a multiple of it."
            )
        x = img_features.flatten(0, 1)

        cls_feat = self.shared_cls(x)
        cls_logits = (
            self.cls(cls_feat).permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.num_classes)
        )
        heatmap_logits = (
            self.heatmap(cls_feat).permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.num_classes)
        )

        reg_feat = self.shared_reg(x)
        bbox_distances = self.bbox(reg_feat).permute(0, 2, 3, 1).contiguous().sigmoid()
        center_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()

        locations = self._token_locations(
            feature_height, feature_width, image_height, image_width
        ).unsqueeze(0)
        bbox_preds = self._decode_bbox_distances(locations, bbox_distances).view(x.shape[0], -1, 4)
        pred_centers2d = self._decode_center_offset(locations, center_offset).view(
            x.shape[0], -1, 2
        )

        # The assigner and the losses take float32; under autocast the convs give half.
        return FocalHead2DOutputs(
            cls_scores=cls_logits.float(),
            bbox_preds=bbox_preds.float(),
            centers_2d=pred_centers2d.float(),
            heatmap=heatmap_logits.float(),
            feature_height=int(feature_height),
            feature_width=int(feature_width),
            image_height=int(image_height),
            image_width=int(image_width),
        )

    def _token_locations(
        self,
        feature_height: int,
        feature_width: int,
        image_height: int,
        image_width: int,
    ) -> Float32[torch.Tensor, "feature_height feature_width 2"]:
        """Build the normalized pixel center of every token on the device of the head.

        Args:
            feature_height: Rows of the token grid.
            feature_width: Columns of the token grid.
            image_height: Image height in pixels.
            image_width: Image width in pixels.

        Returns:
            ``(x, y)`` of every token normalized by the image size.
        """
        device = self.cls.weight.device
        shifts_x = (
            torch.arange(0, image_width, step=self.stride, dtype=torch.float32, device=device)
            + self.stride // 2
        ) / image_width
        shifts_y = (
            torch.arange(0, image_height, step=self.stride, dtype=torch.float32, device=device)
            + self.stride // 2
        ) / image_height
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        return torch.stack((shift_x, shift_y), dim=-1)

    @staticmethod
    def _decode_bbox_distances(
        locations: Float32[torch.Tensor, "1 feature_height feature_width 2"],
        bbox_distances: Float[torch.Tensor, "num_images feature_height feature_width 4"],
    ) -> Float32[torch.Tensor, "num_images feature_height feature_width 4"]:
        """Decode the normalized distances to the box edges into ``(cx, cy, w, h)`` boxes.

        Args:
            locations: Normalized pixel center of every token.
            bbox_distances: Normalized distances of every token to the left, top, right and
                bottom edges of its box, in that order.

        Returns:
            The boxes clamped to the image, normalized by its size.
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

    @staticmethod
    def _decode_center_offset(
        locations: Float32[torch.Tensor, "1 feature_height feature_width 2"],
        center_offset: Float[torch.Tensor, "num_images feature_height feature_width 2"],
    ) -> Float32[torch.Tensor, "num_images feature_height feature_width 2"]:
        """Decode the center offsets in inverse-sigmoid space into normalized centers.

        Args:
            locations: Normalized pixel center of every token.
            center_offset: Offset of every token from its location in inverse-sigmoid space.

        Returns:
            The projected centers normalized by the image size.
        """
        return (inverse_sigmoid(locations) + center_offset).sigmoid()

    def loss(
        self,
        outputs: FocalHead2DOutputs,
        gt_detections_2d: Detection2DGTBatch,
        padding_mask: Bool[torch.Tensor, " batch_size"],
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the five auxiliary 2D losses.

        Args:
            outputs: Dense 2D predictions of the head.
            gt_detections_2d: The collated 2D boxes of every camera of every sample, padded
                per camera, in pixels of the images the predictions are normalized by.
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
            ValueError: If a box or center is not finite or lies outside the image.
        """
        num_images = outputs.cls_scores.shape[0]
        num_cams = num_images // padding_mask.shape[0]
        real_samples = ~padding_mask.view(-1, 1)
        gt_annotated_class_masks = (
            real_samples.expand(-1, self.num_classes)
            if gt_annotated_class_masks is None
            else gt_annotated_class_masks & real_samples
        )
        ground_truth = self._flatten_ground_truth(
            outputs, gt_detections_2d, padding_mask.repeat_interleave(num_cams)
        )
        targets = self._build_targets(outputs, ground_truth)
        heatmap_targets = self._build_heatmaps(
            ground_truth, outputs.feature_height, outputs.feature_width
        )
        # The positive count of the step, averaged over the ranks and floored on its sum (see
        # reduce_mean_count); evaluation averages no gradient, so it uses the count of the rank.
        local_pos = torch.tensor(float(targets.num_pos), device=outputs.cls_scores.device)
        num_total_pos = (
            reduce_mean_count(local_pos) if self.training else local_pos.clamp(min=1.0)
        ).item()

        box_losses, iou_scores = self._box_losses(outputs, targets, num_total_pos)
        losses = self._classification_losses(
            outputs,
            targets,
            heatmap_targets,
            iou_scores,
            gt_annotated_class_masks,
            num_cams,
            num_total_pos,
        )
        losses.update(box_losses)
        return losses

    @staticmethod
    def _flatten_ground_truth(
        outputs: FocalHead2DOutputs,
        gt_detections_2d: Detection2DGTBatch,
        padding_mask: Bool[torch.Tensor, " num_images"],
    ) -> _GroundTruth2D:
        """Lay the collated 2D boxes out one image after the other, as the predictions are.

        Args:
            outputs: Dense 2D predictions, for the device and the image size.
            gt_detections_2d: The collated 2D boxes of every camera of every sample.
            padding_mask: True on the images of the padding samples, whose boxes are left out.

        Returns:
            The boxes, labels and centers of every image, cut to the largest valid count of
            the batch, with the valid mask.

        Raises:
            ValueError: If a valid box or center is not finite or lies outside the image.
        """
        device = outputs.cls_scores.device
        rows = gt_detections_2d.gt_bboxes_2d.flatten(0, 1).to(device=device, dtype=torch.float32)
        labels = gt_detections_2d.gt_labels_2d.flatten(0, 1).to(device=device, dtype=torch.long)
        counts = gt_detections_2d.gt_valid_bboxes_2d.flatten().to(device)
        counts = counts.masked_fill(padding_mask.to(device), 0)
        max_num_boxes = int(counts.max()) if counts.numel() > 0 else 0
        rows, labels = rows[:, :max_num_boxes], labels[:, :max_num_boxes]
        valid_masks = torch.arange(max_num_boxes, device=device) < counts.unsqueeze(1)
        image_extent = rows.new_tensor([outputs.image_width, outputs.image_height]).repeat(3)
        inside = torch.isfinite(rows) & (rows >= 0.0) & (rows <= image_extent)
        if not bool(inside.all(dim=-1)[valid_masks].all()):
            raise ValueError(
                "The 2D boxes or centers hold a non-finite value or one outside the "
                f"{outputs.image_width} x {outputs.image_height} image; the annotation "
                "transform must drop the boxes it cannot project and clip the rest to the image."
            )
        return _GroundTruth2D(
            bboxes=rows[..., Box2DFieldIndex.X_MIN : Box2DFieldIndex.Y_MAX + 1],
            labels=labels.masked_fill(~valid_masks, -1),
            centers=rows[..., Box2DFieldIndex.CENTER_X : Box2DFieldIndex.CENTER_Y + 1],
            valid_masks=valid_masks,
        )

    def _build_targets(
        self, outputs: FocalHead2DOutputs, ground_truth: _GroundTruth2D
    ) -> _Targets2D:
        """Match the tokens of every image to its boxes and build the targets of every token.

        The assigner computes the costs of the batch in one call and runs the Hungarian solver
        per image on the host; no gradient goes through the assignment.

        Args:
            outputs: Dense 2D predictions of the head.
            ground_truth: Padded ground truth of every image.

        Returns:
            The targets of every token, concatenated over the images.
        """
        image_height, image_width = outputs.image_height, outputs.image_width
        num_images, num_tokens = outputs.cls_scores.shape[:2]
        if ground_truth.bboxes.shape[1] == 0:
            return _Targets2D(
                labels=torch.full(
                    (num_images * num_tokens,), self.num_classes, device=outputs.cls_scores.device
                ),
                is_positive=torch.zeros(
                    num_images * num_tokens, dtype=torch.bool, device=outputs.cls_scores.device
                ),
                bbox_targets=outputs.bbox_preds.new_zeros((num_images * num_tokens, 4)),
                centers2d_targets=outputs.bbox_preds.new_zeros((num_images * num_tokens, 2)),
                num_pos=0,
            )
        assigned = self.assigner.assign(
            outputs.bbox_preds.detach(),
            outputs.cls_scores.detach(),
            outputs.centers_2d.detach(),
            ground_truth.bboxes,
            ground_truth.labels,
            ground_truth.centers,
            ground_truth.valid_masks,
            image_height,
            image_width,
        )
        is_positive = assigned.gt_inds > 0
        matched = (assigned.gt_inds - 1).clamp(min=0)
        labels = torch.where(is_positive, assigned.labels, self.num_classes)
        factor = ground_truth.bboxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        matched_boxes = ground_truth.bboxes.gather(1, matched.unsqueeze(-1).expand(-1, -1, 4))
        bbox_targets = BaseBBoxes2D(matched_boxes.reshape(-1, 4) / factor).cxcywh
        matched_centers = ground_truth.centers.gather(1, matched.unsqueeze(-1).expand(-1, -1, 2))
        centers2d_targets = matched_centers.reshape(-1, 2) / factor[:2]
        positive = is_positive.reshape(-1, 1)
        return _Targets2D(
            labels=labels.reshape(-1),
            is_positive=is_positive.reshape(-1),
            bbox_targets=bbox_targets * positive,
            centers2d_targets=centers2d_targets * positive,
            num_pos=int(is_positive.sum()),
        )

    def _build_heatmaps(
        self,
        ground_truth: _GroundTruth2D,
        feature_height: int,
        feature_width: int,
    ) -> Float32[torch.Tensor, "num_images num_tokens num_classes"]:
        """Draw the center heatmap of every image on the token grid, one cell per token.

        Every box peaks in the channel of its class with a Gaussian whose radius is its
        smallest distance from the center to an edge, in tokens. A center on the right or
        bottom edge of the image, where the annotation transform clamps a center beyond it,
        peaks in the last cell, as a center on the left or top edge peaks in cell 0.

        Args:
            ground_truth: Padded ground truth of every image.
            feature_height: Rows of the token grid.
            feature_width: Columns of the token grid.

        Returns:
            The heatmaps flattened to one row per token, as the heatmap logits.
        """
        device = ground_truth.bboxes.device
        num_images = ground_truth.bboxes.shape[0]
        if ground_truth.bboxes.shape[1] == 0:
            return torch.zeros(
                num_images, feature_height * feature_width, self.num_classes, device=device
            )
        edge_distances = torch.cat(
            [
                ground_truth.centers - ground_truth.bboxes[..., :2],
                ground_truth.bboxes[..., 2:] - ground_truth.centers,
            ],
            dim=-1,
        )
        radii = torch.ceil(edge_distances.amin(dim=-1) / self.stride).clamp(min=1.0).int()
        grid_max = torch.tensor([feature_width - 1, feature_height - 1], device=device)
        cells = torch.minimum((ground_truth.centers / self.stride).long(), grid_max)
        heatmaps = create_gaussian_heatmaps(
            feature_width,
            feature_height,
            self.num_classes,
            cells,
            radii,
            ground_truth.labels,
            ground_truth.valid_masks,
            device,
        )
        # (num_images, num_classes, H, W) -> (num_images, H * W, num_classes), as the logits.
        return heatmaps.flatten(2).transpose(1, 2)

    def _box_losses(
        self, outputs: FocalHead2DOutputs, targets: _Targets2D, num_total_pos: float
    ) -> tuple[dict[str, Float[torch.Tensor, ""]], Float32[torch.Tensor, " num_tokens"]]:
        """Compute the box L1, GIoU and center L1 losses of the matched tokens.

        Args:
            outputs: Dense 2D predictions of the head.
            targets: Targets of every token.
            num_total_pos: Positive count the losses are divided by.

        Returns:
            The three losses, and the detached IoU of every token with its target box, the
            quality target of the classification.
        """
        image_height, image_width = outputs.image_height, outputs.image_width
        positive_weights = targets.is_positive.to(outputs.bbox_preds.dtype)
        factor = outputs.bbox_preds.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        flat_bbox_preds = outputs.bbox_preds.reshape(-1, 4)
        pred_boxes = BaseBBoxes2D(BaseBBoxes2D.from_cxcywh(flat_bbox_preds).bbox_params * factor)
        target_boxes = BaseBBoxes2D(
            BaseBBoxes2D.from_cxcywh(targets.bbox_targets).bbox_params * factor
        )
        loss_iou = (
            self.loss_iou_weight
            * ((1.0 - pred_boxes.giou(target_boxes, is_aligned=True)) * positive_weights).sum()
            / num_total_pos
        )
        loss_bbox = (
            self.loss_bbox_weight
            * (
                self.loss_l1(flat_bbox_preds, targets.bbox_targets).sum(dim=-1) * positive_weights
            ).sum()
            / num_total_pos
        )
        loss_centers2d = (
            self.loss_centers2d_weight
            * (
                self.loss_l1(outputs.centers_2d.reshape(-1, 2), targets.centers2d_targets).sum(
                    dim=-1
                )
                * positive_weights
            ).sum()
            / num_total_pos
        )
        losses = {
            "loss_bbox2d": loss_bbox,
            "loss_iou2d": loss_iou,
            "loss_centers2d": loss_centers2d,
        }
        # The IoU is the classification target, not a gradient path for the box.
        return losses, target_boxes.iou(pred_boxes, is_aligned=True).detach()

    def _classification_losses(
        self,
        outputs: FocalHead2DOutputs,
        targets: _Targets2D,
        heatmap_targets: Float32[torch.Tensor, "num_images num_tokens num_classes"],
        iou_scores: Float32[torch.Tensor, " num_tokens"],
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"],
        num_cams: int,
        num_total_pos: float,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the quality focal classification loss and the heatmap loss.

        Args:
            outputs: Dense 2D predictions of the head.
            targets: Targets of every token.
            heatmap_targets: Heatmap of every image, one row per token.
            iou_scores: Detached IoU of every token with its target box, the quality target.
            gt_annotated_class_masks: True where a class is annotated in a sample, False on
                every class of a padding sample.
            num_cams: Images per sample.
            num_total_pos: Positive count the losses are divided by.

        Returns:
            The two losses.
        """
        cls_scores = outputs.cls_scores
        class_weights = self._class_weights(
            gt_annotated_class_masks, num_cams, cls_scores.size(1), cls_scores.dtype
        )
        # A token matched to a box keeps every term.
        class_weights[targets.is_positive] = 1.0
        loss_cls = self.loss_cls_weight * self.loss_cls(
            cls_scores.reshape(-1, self.num_classes),
            targets.labels,
            iou_scores,
            num_total_pos,
            weights=class_weights,
        )
        heatmap_weights = self._class_weights(
            gt_annotated_class_masks, num_cams, 1, cls_scores.dtype
        ).view(cls_scores.size(0), 1, self.num_classes)
        loss_heatmap = self.loss_heatmap_weight * self.loss_heatmap(
            outputs.heatmap, heatmap_targets, weights=heatmap_weights, avg_factor=num_total_pos
        )
        return {"loss_cls2d": loss_cls, "loss_heatmap2d": loss_heatmap}

    def _class_weights(
        self,
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"],
        num_cams: int,
        tokens_per_image: int,
        dtype: torch.dtype,
    ) -> Float[torch.Tensor, "num_tokens num_classes"]:
        """Broadcast the annotated class mask of every sample to the tokens of its images.

        Args:
            gt_annotated_class_masks: True where a class is annotated in a sample.
            num_cams: Images per sample, which follow each other.
            tokens_per_image: Tokens the mask row is repeated over per image.
            dtype: Dtype of the weights.

        Returns:
            One weight row per token of every image, 1 where the class is annotated.
        """
        per_image = gt_annotated_class_masks.repeat_interleave(num_cams, dim=0)
        return (
            per_image.unsqueeze(1)
            .expand(-1, tokens_per_image, -1)
            .reshape(-1, self.num_classes)
            .to(dtype=dtype)
        )
