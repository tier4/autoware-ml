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

"""Rotated BEV IoU loss between matched pairs of predicted and ground-truth boxes."""

from __future__ import annotations

from typing import Sequence

from jaxtyping import Bool, Float32, Int64
import torch
import torch.nn as nn

from autoware_ml.ops.diff_iou_rotated.diff_iou_rotated import oriented_box_intersection_2d


class Rotated2DIouLoss(nn.Module):
    """Compute ``1 - IoU`` of the rotated BEV footprints of matched box pairs.

    Both inputs use the encoded box layout ``TransFusionBBoxCoder.encode`` produces: BEV
    center in feature-grid cells, log sizes, and the yaw as a (sin, cos) pair. The loss decodes
    them into metric BEV corners itself, so it needs the same grid geometry as the coder, and
    computes the polygon intersection with the differentiable rotated IoU op.

    Classes listed in ``labels_to_ignore_rotation`` (for example traffic cones, whose yaw is
    undefined) are compared axis-aligned, so their orientation never enters the IoU.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        out_size_factor: int,
        labels_to_ignore_rotation: Sequence[str] | None = None,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the rotated 2D IoU loss.

        Args:
            class_names: All class names, in label-id order.
            point_cloud_range: Detector range ``[x_min, y_min, z_min, x_max, y_max, z_max]``;
                only the BEV origin is used to decode centers.
            voxel_size: Voxel size along each axis; only the BEV sizes are used.
            out_size_factor: BEV downsampling factor between the voxel grid and the head's
                feature map, so ``voxel_size * out_size_factor`` is the size of one grid cell.
            labels_to_ignore_rotation: Class names whose boxes are compared axis-aligned.
            loss_weight: Weight applied to the loss.

        Raises:
            ValueError: If a name in ``labels_to_ignore_rotation`` is not in ``class_names``.
        """
        super().__init__()
        self.class_names = list(class_names)
        self.point_cloud_range = list(point_cloud_range)
        self.voxel_size = list(voxel_size)
        self.out_size_factor = out_size_factor
        self.labels_to_ignore_rotation = (
            list(labels_to_ignore_rotation) if labels_to_ignore_rotation is not None else []
        )
        self.labels_id_to_ignore_rotation = self._resolve_ignore_label_ids()
        self.loss_weight = loss_weight

    def _resolve_ignore_label_ids(self) -> list[int]:
        """Resolve ``labels_to_ignore_rotation`` into label ids."""
        unknown = [name for name in self.labels_to_ignore_rotation if name not in self.class_names]
        if unknown:
            raise ValueError(
                f"labels_to_ignore_rotation {unknown} are not in class_names {self.class_names}."
            )
        return [self.class_names.index(name) for name in self.labels_to_ignore_rotation]

    def convert_to_bev_corners(
        self,
        bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        labels: Int64[torch.Tensor, "batch_size num_bboxes"],
        is_gt: bool = False,
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_bboxes 4 2"],
        Float32[torch.Tensor, "batch_size num_bboxes 2"],
    ]:
        """Decode encoded boxes into metric BEV corners.

        Predictions are decoded like targets, except that their (sin, cos) pair is first
        normalized to unit length so the corners are rotated by a proper rotation. Boxes of the
        classes in ``labels_id_to_ignore_rotation`` are laid out axis-aligned.

        Args:
            bboxes: Encoded boxes; only the first eight channels are read.
            labels: Class label per box.
            is_gt: Whether ``bboxes`` are targets, whose (sin, cos) already has unit norm.

        Returns:
            Corners as (top right, top left, bottom left, bottom right) in metric BEV
            coordinates, and the (length, width) of every box.
        """
        batch_size = bboxes.shape[0]
        cell_size_x = self.out_size_factor * self.voxel_size[0]
        cell_size_y = self.out_size_factor * self.voxel_size[1]
        center_x = bboxes[:, :, 0] * cell_size_x + self.point_cloud_range[0]
        center_y = bboxes[:, :, 1] * cell_size_y + self.point_cloud_range[1]
        # (batch_size, num_bboxes, 2) as (length, width)
        lw = bboxes[:, :, 3:5].exp()
        rot_sin = bboxes[:, :, 6:7]
        rot_cos = bboxes[:, :, 7:8]
        if not is_gt:
            norm_rotation = torch.sqrt(rot_sin.square() + rot_cos.square() + 1e-6)
            rot_sin = rot_sin / norm_rotation
            rot_cos = rot_cos / norm_rotation

        # Row-vector convention, corners @ R^T, matching box2corners in the diff IoU op.
        row1 = torch.cat([rot_cos, rot_sin], dim=-1)
        row2 = torch.cat([-rot_sin, rot_cos], dim=-1)  # (batch_size, num_bboxes, 2)
        rotation_matrix_transpose = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)

        if self.labels_id_to_ignore_rotation:
            # (batch_size, num_bboxes, 1, 1)
            ignore_masks = torch.isin(labels, labels.new_tensor(self.labels_id_to_ignore_rotation))[
                ..., None, None
            ]
            identity = torch.eye(
                2,
                device=rotation_matrix_transpose.device,
                dtype=rotation_matrix_transpose.dtype,
            ).view(1, 1, 2, 2)
            rotation_matrix_transpose = torch.where(
                ignore_masks, identity, rotation_matrix_transpose
            )

        x4 = lw.new_tensor([0.5, -0.5, -0.5, 0.5]) * lw[:, :, 0].unsqueeze(-1)  # (B, N, 4)
        y4 = lw.new_tensor([0.5, 0.5, -0.5, -0.5]) * lw[:, :, 1].unsqueeze(-1)  # (B, N, 4)
        # (top right, top left, bottom left, bottom right), (batch_size, num_bboxes, 4, 2)
        corners = torch.stack([x4, y4], dim=-1)

        # (B * N, 4, 2) @ (B * N, 2, 2) -> (B * N, 4, 2) -> (B, N, 4, 2)
        rotated = torch.bmm(
            corners.reshape(-1, 4, 2), rotation_matrix_transpose.reshape(-1, 2, 2)
        ).view(batch_size, -1, 4, 2)
        translation = torch.stack([center_x, center_y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
        return rotated + translation, lw

    def forward(
        self,
        prediction_bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        target_bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"],
        labels: Int64[torch.Tensor, "batch_size num_bboxes"],
        bbox_weights: Float32[torch.Tensor, "batch_size num_bboxes"]
        | Bool[torch.Tensor, "batch_size num_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes"]:
        """Compute the weighted ``1 - IoU`` of every (prediction, target) pair.

        Args:
            prediction_bboxes: Encoded predicted boxes.
            target_bboxes: Encoded target boxes, paired element-wise with the predictions.
            labels: Class label per pair, used for the rotation-ignored classes.
            bbox_weights: Weight per pair. A zero drops the pair from the loss, for example an
                unmatched proposal.

        Returns:
            Weighted loss for each pair.
        """
        prediction_corners, prediction_dims = self.convert_to_bev_corners(
            prediction_bboxes, labels, is_gt=False
        )
        target_corners, target_dims = self.convert_to_bev_corners(target_bboxes, labels, is_gt=True)
        intersection, _ = oriented_box_intersection_2d(prediction_corners, target_corners)
        area1 = prediction_dims[:, :, 0] * prediction_dims[:, :, 1]
        area2 = target_dims[:, :, 0] * target_dims[:, :, 1]
        union = area1 + area2 - intersection
        ious = intersection / (union + 1e-8)
        return self.loss_weight * bbox_weights.to(ious.dtype) * (1.0 - ious)
