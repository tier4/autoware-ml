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

"""Matching costs of the 2D Hungarian assigner.

Each cost compares the proposals of a batch of images with the ground truth of the same images,
padded to one count, and gives one ``(num_queries, max_num_gts)`` block per image.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float32
import torch

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D


@dataclass(frozen=True)
class BBoxL1Cost2D:
    """Pairwise L1 cost between normalized 2D boxes of a batch of images.

    Predictions are ``(cx, cy, w, h)`` in [0, 1]; ground truth arrives as
    normalized ``(x1, y1, x2, y2)`` and is converted according to
    ``box_format``.
    """

    weight: float = 1.0
    box_format: str = "xywh"

    def __post_init__(self) -> None:
        """Refuse a box form other than ``xywh`` and ``xyxy``.

        Raises:
            ValueError: If ``box_format`` is another form.
        """
        if self.box_format not in ("xywh", "xyxy"):
            raise ValueError(f"Unsupported box_format '{self.box_format}'.")

    def __call__(
        self,
        bbox_pred: Float32[torch.Tensor, "batch_size num_queries 4"],
        gt_bboxes: Float32[torch.Tensor, "batch_size num_gts 4"],
    ) -> Float32[torch.Tensor, "batch_size num_queries num_gts"]:
        """Compute the pairwise box L1 cost of every image.

        Args:
            bbox_pred: Normalized ``(cx, cy, w, h)`` proposals of every image.
            gt_bboxes: Normalized ``(x1, y1, x2, y2)`` boxes of every image, padded to the
                same count.

        Returns:
            The weighted L1 distance of every pair of an image.
        """
        if self.box_format == "xywh":
            gt_bboxes = BaseBBoxes2D(gt_bboxes.reshape(-1, 4)).cxcywh.view(gt_bboxes.shape)
        else:
            bbox_pred = BaseBBoxes2D.from_cxcywh(bbox_pred.reshape(-1, 4)).bbox_params.view(
                bbox_pred.shape
            )
        return torch.cdist(bbox_pred, gt_bboxes, p=1) * self.weight


@dataclass(frozen=True)
class IoUCost2D:
    """Pairwise negative-IoU (or GIoU) cost between unnormalized 2D boxes of a batch of images."""

    weight: float = 1.0
    iou_mode: str = "giou"

    def __post_init__(self) -> None:
        """Refuse an overlap mode other than ``iou`` and ``giou``.

        Raises:
            ValueError: If ``iou_mode`` is another mode.
        """
        if self.iou_mode not in ("iou", "giou"):
            raise ValueError(f"Unsupported iou_mode '{self.iou_mode}'.")

    def __call__(
        self,
        bboxes: Float32[torch.Tensor, "batch_size num_queries 4"],
        gt_bboxes: Float32[torch.Tensor, "batch_size num_gts 4"],
    ) -> Float32[torch.Tensor, "batch_size num_queries num_gts"]:
        """Compute the pairwise overlap cost of every image.

        Args:
            bboxes: ``(x1, y1, x2, y2)`` proposals of every image in pixels.
            gt_bboxes: ``(x1, y1, x2, y2)`` boxes of every image in pixels, padded to the same
                count.

        Returns:
            The weighted negative overlap of every pair of an image.
        """
        return -self._pairwise_overlap(bboxes, gt_bboxes, self.iou_mode == "giou") * self.weight

    @staticmethod
    def _pairwise_overlap(
        boxes: Float32[torch.Tensor, "batch_size num_queries 4"],
        gt_bboxes: Float32[torch.Tensor, "batch_size num_gts 4"],
        generalized: bool,
        eps: float = 1e-6,
    ) -> Float32[torch.Tensor, "batch_size num_queries num_gts"]:
        """Compute the IoU, or the GIoU, of every proposal with every box of its image.

        The batched form of :meth:`BaseBBoxes2D.iou` and :meth:`BaseBBoxes2D.giou`, with the same
        floors, so the matching costs of a batch of images are computed in one call.

        Args:
            boxes: ``(x1, y1, x2, y2)`` proposals of every image.
            gt_bboxes: ``(x1, y1, x2, y2)`` boxes of every image, padded to the same count.
            generalized: Whether to compute the GIoU instead of the IoU.
            eps: Floor of the union and of the enclosing area, so a zero-area pair gives 0.

        Returns:
            The overlap of every pair of an image.
        """
        boxes1, boxes2 = boxes.unsqueeze(2), gt_bboxes.unsqueeze(1)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        wh = (
            torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
            - torch.maximum(boxes1[..., :2], boxes2[..., :2])
        ).clamp(min=0)
        intersection = wh[..., 0] * wh[..., 1]
        union = (area1 + area2 - intersection).clamp(min=eps)
        iou = intersection / union
        if not generalized:
            return iou
        enclosing_wh = (
            torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
            - torch.minimum(boxes1[..., :2], boxes2[..., :2])
        ).clamp(min=0)
        enclosing_area = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp(min=eps)
        return iou - (enclosing_area - union) / enclosing_area


@dataclass(frozen=True)
class Center2DL1Cost:
    """Pairwise L1 cost between normalized projected 2D centers of a batch of images."""

    weight: float = 1.0

    def __call__(
        self,
        center_pred: Float32[torch.Tensor, "batch_size num_queries 2"],
        gt_centers: Float32[torch.Tensor, "batch_size num_gts 2"],
    ) -> Float32[torch.Tensor, "batch_size num_queries num_gts"]:
        """Compute the pairwise center L1 cost of every image.

        Args:
            center_pred: Normalized projected centers of the proposals of every image.
            gt_centers: Normalized projected centers of the boxes of every image, padded to
                the same count.

        Returns:
            The weighted L1 distance of every pair of an image.
        """
        return torch.cdist(center_pred, gt_centers, p=1) * self.weight
