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

"""2D bounding-box utilities for auxiliary image-plane supervision.

These helpers back the auxiliary 2D detection head used by camera 3D
detectors (Focal-PETR-style): box format conversion and IoU/GIoU overlaps.
"""

from __future__ import annotations

import torch


def bbox_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def bbox_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(cx, cy, w, h)``."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def bbox_overlaps(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    mode: str = "iou",
    is_aligned: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute IoU or GIoU between two sets of ``(x1, y1, x2, y2)`` boxes.

    Args:
        boxes1: Boxes with shape ``(N, 4)``.
        boxes2: Boxes with shape ``(M, 4)`` (or ``(N, 4)`` when aligned).
        mode: ``"iou"`` or ``"giou"``.
        is_aligned: Compute element-wise overlaps instead of pairwise.
        eps: Numerical stability constant.

    Returns:
        Overlap tensor with shape ``(N,)`` when aligned, else ``(N, M)``.
    """
    if mode not in ("iou", "giou"):
        raise ValueError(f"Unsupported overlap mode '{mode}'.")
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (
        boxes1[..., 3] - boxes1[..., 1]
    ).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (
        boxes2[..., 3] - boxes2[..., 1]
    ).clamp(min=0)

    if not is_aligned:
        boxes1 = boxes1[:, None, :]
        boxes2 = boxes2[None, :, :]
        area1 = area1[:, None]
        area2 = area2[None, :]

    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = (area1 + area2 - intersection).clamp(min=eps)
    ious = intersection / union
    if mode == "iou":
        return ious

    enclose_lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = (enclose_wh[..., 0] * enclose_wh[..., 1]).clamp(min=eps)
    return ious - (enclose_area - union) / enclose_area
