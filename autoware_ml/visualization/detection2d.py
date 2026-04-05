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

"""Reusable 2D detection visualization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import torch
import torchvision


def build_label_names(
    categories: Sequence[Mapping[str, Any]] | None,
    *,
    num_classes: int | None = None,
) -> dict[int, str]:
    """Build a contiguous label-id to display-name mapping."""
    if categories and (num_classes is None or len(categories) == num_classes):
        return {
            index: str(category.get("name", f"class_{index}"))
            for index, category in enumerate(categories)
        }
    total = num_classes if num_classes is not None else len(categories or [])
    return {index: f"class_{index}" for index in range(total)}


def targets_to_absolute_xyxy(
    target: Mapping[str, Any],
    orig_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized ``cxcywh`` targets into absolute ``xyxy`` boxes."""
    boxes = target["boxes"]
    labels = target["labels"]
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4), labels

    scale = orig_size[[1, 0, 1, 0]].to(device=boxes.device, dtype=boxes.dtype)
    absolute_boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy") * scale
    return absolute_boxes, labels


def _draw_box(
    image_bgr: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    text: str | None,
    line_thickness: int,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    text_top = max(0, y1 - text_height - baseline - 4)
    cv2.rectangle(
        image_bgr,
        (x1, text_top),
        (x1 + text_width + 4, text_top + text_height + baseline + 4),
        color,
        thickness=-1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x1 + 2, text_top + text_height + 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_detection_preview(
    image_bgr: np.ndarray,
    prediction: Mapping[str, torch.Tensor],
    *,
    label_names: Mapping[int, str] | None = None,
    score_threshold: float = 0.25,
    line_thickness: int = 2,
    ground_truth: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> np.ndarray:
    """Draw model predictions and optional ground truth on an image."""
    preview = image_bgr.copy()
    label_names = label_names or {}

    if ground_truth is not None:
        gt_boxes, gt_labels = ground_truth
        for box, label in zip(gt_boxes.cpu(), gt_labels.cpu(), strict=False):
            _draw_box(
                preview,
                box.numpy(),
                color=(255, 128, 0),
                text=f"gt:{label_names.get(int(label), f'class_{int(label)}')}",
                line_thickness=max(1, line_thickness - 1),
            )

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    keep = scores >= score_threshold

    for box, score, label in zip(boxes[keep], scores[keep], labels[keep], strict=False):
        label_text = label_names.get(int(label), f"class_{int(label)}")
        _draw_box(
            preview,
            box.numpy(),
            color=(0, 200, 0),
            text=f"{label_text}:{float(score):.2f}",
            line_thickness=line_thickness,
        )

    return preview
