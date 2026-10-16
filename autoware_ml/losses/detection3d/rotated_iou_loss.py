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

from autoware_ml.ops.diff_iou_rotated.diff_iou_rotated import diff_iou_rotated_2d
from autoware_ml.types.geometry import Box3DFieldIndex


class Rotated2DIouLoss(nn.Module):
    """Compute ``1 - IoU`` of the rotated BEV footprints of matched box pairs.

    Both inputs are metric boxes in the :class:`~autoware_ml.types.geometry.Box3DFieldIndex`
    layout, as ``TransFusionBBoxCoder.decode_boxes`` produces them. The caller decodes the
    regression channels with its own box coder, so the loss shares the coder's grid geometry
    by construction instead of carrying a copy of it. The IoU comes from
    :func:`~autoware_ml.ops.diff_iou_rotated.diff_iou_rotated.diff_iou_rotated_2d`, which
    runs the polygon intersection in float32 whatever the input dtype.

    Classes listed in ``labels_to_ignore_rotation`` (for example traffic cones, whose yaw is
    undefined) are compared axis-aligned, so their orientation never enters the IoU.
    """

    _BEV_FIELDS = [
        Box3DFieldIndex.X,
        Box3DFieldIndex.Y,
        Box3DFieldIndex.LENGTH,
        Box3DFieldIndex.WIDTH,
        Box3DFieldIndex.YAW,
    ]

    def __init__(
        self,
        class_names: Sequence[str],
        labels_to_ignore_rotation: Sequence[str] | None = None,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the rotated 2D IoU loss.

        Args:
            class_names: All class names, in label-id order.
            labels_to_ignore_rotation: Class names whose boxes are compared axis-aligned.
            loss_weight: Weight applied to the loss.

        Raises:
            ValueError: If a name in ``labels_to_ignore_rotation`` is not in ``class_names``.
        """
        super().__init__()
        self.class_names = list(class_names)
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

    def _bev_boxes(
        self,
        bboxes: Float32[torch.Tensor, "batch_size num_bboxes box_size"],
        labels: Int64[torch.Tensor, "batch_size num_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes 5"]:
        """Pick the BEV footprint of every box, axis-aligned for the rotation-ignored classes.

        Args:
            bboxes: Metric boxes in the ``Box3DFieldIndex`` layout.
            labels: Class label per box.

        Returns:
            Boxes as (x, y, length, width, yaw), the layout ``diff_iou_rotated_2d`` takes.
        """
        bev = bboxes[..., self._BEV_FIELDS]
        if not self.labels_id_to_ignore_rotation:
            return bev
        ignore_masks = torch.isin(labels, labels.new_tensor(self.labels_id_to_ignore_rotation))
        yaw = bev[..., 4].masked_fill(ignore_masks, 0.0)
        return torch.cat([bev[..., :4], yaw.unsqueeze(-1)], dim=-1)

    def forward(
        self,
        prediction_bboxes: Float32[torch.Tensor, "batch_size num_bboxes box_size"],
        target_bboxes: Float32[torch.Tensor, "batch_size num_bboxes box_size"],
        labels: Int64[torch.Tensor, "batch_size num_bboxes"],
        bbox_weights: Float32[torch.Tensor, "batch_size num_bboxes"]
        | Bool[torch.Tensor, "batch_size num_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_bboxes"]:
        """Compute the weighted ``1 - IoU`` of every (prediction, target) pair.

        Args:
            prediction_bboxes: Decoded predicted boxes in the ``Box3DFieldIndex`` layout. Only
                the BEV center, length, width and yaw are read.
            target_bboxes: Decoded target boxes in the same layout, paired element-wise with
                the predictions.
            labels: Class label per pair, used for the rotation-ignored classes.
            bbox_weights: Weight per pair. A zero drops the pair from the loss, for example an
                unmatched proposal.

        Returns:
            Weighted float32 loss for each pair.
        """
        ious = diff_iou_rotated_2d(
            self._bev_boxes(prediction_bboxes, labels), self._bev_boxes(target_bboxes, labels)
        )
        return self.loss_weight * bbox_weights.to(ious.dtype) * (1.0 - ious)
