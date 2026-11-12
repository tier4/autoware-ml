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

"""Axis-aligned 2D boxes on the image plane.

The auxiliary 2D head of camera 3D detectors (Focal-PETR-style) matches and supervises its
proposals with these boxes: box format conversion and IoU / GIoU overlaps.

torchvision ships ``box_convert``, ``box_iou`` and ``generalized_box_iou``, but none of them
computes aligned (element-wise) overlaps, and without an ``eps`` a zero-area pair gives NaN
where the matching cost needs 0. The overlaps here are tested against torchvision's.
"""

from __future__ import annotations

from jaxtyping import Float32
import torch
from torch import Tensor


class BaseBBoxes2D:
    """Axis-aligned 2D boxes stored as ``(x1, y1, x2, y2)``, in pixels or normalized.

    Unlike :class:`~autoware_ml.geometry.bbox_3d.base_bbox3d.BaseBBoxes3D`, the boxes carry no
    labels, since the 2D head builds them from its proposals as well as from the ground truth.

    The boxes are float32: the areas of pixel boxes overflow float16 (a 1900 x 1000 box is
    1.9e6, above the float16 maximum of 65504), and a bfloat16 box is off by up to 1/256 of the
    image. Callers running under autocast cast their outputs to float32 first.
    """

    def __init__(self, bbox_params: Float32[Tensor, "num_bboxes 4"]) -> None:
        """
        Args:
            bbox_params: ``(x1, y1, x2, y2)`` of every box.

        Raises:
            ValueError: If ``bbox_params`` is not a float32 ``(num_bboxes, 4)`` tensor.
        """
        if bbox_params.ndim != 2 or bbox_params.shape[1] != 4:
            raise ValueError(
                f"bbox_params must have shape (num_bboxes, 4), got {tuple(bbox_params.shape)}."
            )
        if bbox_params.dtype != torch.float32:
            raise ValueError(f"bbox_params must be float32, got {bbox_params.dtype}.")
        self._bbox_params = bbox_params

    @classmethod
    def from_cxcywh(cls, boxes: Float32[Tensor, "num_bboxes 4"]) -> BaseBBoxes2D:
        """Build the boxes from their ``(cx, cy, w, h)``."""
        cx, cy, w, h = boxes.unbind(-1)
        return cls(torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1))

    def __len__(self) -> int:
        """Get the number of boxes."""
        return self._bbox_params.shape[0]

    @property
    def bbox_params(self) -> Float32[Tensor, "num_bboxes 4"]:
        """``(x1, y1, x2, y2)`` of every box."""
        return self._bbox_params

    @property
    def cxcywh(self) -> Float32[Tensor, "num_bboxes 4"]:
        """``(cx, cy, w, h)`` of every box."""
        x1, y1, x2, y2 = self._bbox_params.unbind(-1)
        return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)

    @property
    def area(self) -> Float32[Tensor, " num_bboxes"]:
        """Area of every box, 0 for a box whose corners are swapped."""
        x1, y1, x2, y2 = self._bbox_params.unbind(-1)
        return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    def iou(
        self, other: BaseBBoxes2D, is_aligned: bool = False, eps: float = 1e-6
    ) -> Float32[Tensor, "num_bboxes num_other"] | Float32[Tensor, " num_bboxes"]:
        """Compute the IoU with ``other``.

        Args:
            other: Boxes to compare with.
            is_aligned: Compare box ``i`` with box ``i`` of ``other`` only, instead of with
                every box of ``other``.
            eps: Floor of the union, so a pair of zero-area boxes gives 0.

        Returns:
            ``(len(self), len(other))`` IoUs, or ``(len(self),)`` when aligned.

        Raises:
            ValueError: If aligned and the two hold different numbers of boxes.
        """
        intersection, union = self._intersection_and_union(other, is_aligned, eps)
        return intersection / union

    def giou(
        self, other: BaseBBoxes2D, is_aligned: bool = False, eps: float = 1e-6
    ) -> Float32[Tensor, "num_bboxes num_other"] | Float32[Tensor, " num_bboxes"]:
        """Compute the generalized IoU with ``other``.

        The IoU minus the share of the smallest enclosing box that the union leaves empty, so
        disjoint boxes get a negative value that grows with their distance.

        Args:
            other: Boxes to compare with.
            is_aligned: Compare box ``i`` with box ``i`` of ``other`` only, instead of with
                every box of ``other``.
            eps: Floor of the union and of the enclosing area, so a pair of zero-area boxes
                gives 0.

        Returns:
            ``(len(self), len(other))`` GIoUs, or ``(len(self),)`` when aligned.

        Raises:
            ValueError: If aligned and the two hold different numbers of boxes.
        """
        intersection, union = self._intersection_and_union(other, is_aligned, eps)
        boxes1, boxes2 = self._paired_bbox_params(other, is_aligned)
        enclosing_lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
        enclosing_rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
        enclosing_area = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp(min=eps)
        return intersection / union - (enclosing_area - union) / enclosing_area

    def _paired_bbox_params(
        self, other: BaseBBoxes2D, is_aligned: bool
    ) -> tuple[Float32[Tensor, "... 4"], Float32[Tensor, "... 4"]]:
        """Get the two sets of boxes broadcast against each other, pair by pair when aligned."""
        if not is_aligned:
            return self._bbox_params[:, None, :], other.bbox_params[None, :, :]
        if len(self) != len(other):
            raise ValueError(
                f"Aligned overlaps need as many boxes on both sides, got {len(self)} and "
                f"{len(other)}."
            )
        return self._bbox_params, other.bbox_params

    def _intersection_and_union(
        self, other: BaseBBoxes2D, is_aligned: bool, eps: float
    ) -> tuple[Float32[Tensor, "..."], Float32[Tensor, "..."]]:
        """Get the intersection and the union, floored at ``eps``, of every pair."""
        boxes1, boxes2 = self._paired_bbox_params(other, is_aligned)
        area1, area2 = self.area, other.area
        if not is_aligned:
            area1, area2 = area1[:, None], area2[None, :]
        lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
        rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[..., 0] * wh[..., 1]
        union = (area1 + area2 - intersection).clamp(min=eps)
        return intersection, union
