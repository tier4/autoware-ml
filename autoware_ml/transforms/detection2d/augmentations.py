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

"""Bbox-aware image augmentations for detection2d pipelines."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image
import torch
from torchvision.transforms import v2 as T

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.detection2d.utils import (
    clone_target,
    convert_boxes_to_tv_tensor,
    resolve_current_epoch,
)


class DetectionTransform(BaseTransform):
    """Base class for dict-based detection transforms with optional epoch scheduling."""

    _required_keys = ["image", "target"]

    def __init__(
        self,
        p: float = 1.0,
        start_epoch: int = 0,
        stop_epoch: int | None = None,
    ) -> None:
        self.p = p
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch

    def __call__(self, input_dict: dict[str, Any], context: Any = None) -> dict[str, Any]:
        self._context = context
        self._validate_required_keys(input_dict)
        self._handle_optional_keys(input_dict)
        current_epoch = resolve_current_epoch(context)
        if current_epoch < self.start_epoch:
            return self.on_skip(input_dict)
        if self.stop_epoch is not None and current_epoch >= self.stop_epoch:
            return self.on_skip(input_dict)
        if not self._should_apply():
            return self.on_skip(input_dict)
        return self.transform(input_dict)

    def _apply_torchvision_transform(self, input_dict: dict[str, Any], transform: Any) -> dict[str, Any]:
        image = input_dict["image"]
        target = clone_target(input_dict["target"])
        image, target = transform(image, target)
        return {"image": image, "target": target}


class RandomPhotometricDistort(DetectionTransform):
    """Apply photometric distortion with torchvision's detection-aware op."""

    def __init__(
        self,
        p: float = 0.5,
        stop_epoch: int | None = None,
    ) -> None:
        super().__init__(p=p, stop_epoch=stop_epoch)
        self.transform_op = T.RandomPhotometricDistort(p=1.0)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        return self._apply_torchvision_transform(input_dict, self.transform_op)


class RandomZoomOut(DetectionTransform):
    """Pad the image canvas before cropping to emulate RT-DETR zoom-out augmentation."""

    def __init__(
        self,
        fill: int = 0,
        p: float = 1.0,
        stop_epoch: int | None = None,
    ) -> None:
        super().__init__(p=p, stop_epoch=stop_epoch)
        self.transform_op = T.RandomZoomOut(fill=fill)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        return self._apply_torchvision_transform(input_dict, self.transform_op)


class RandomHorizontalFlip(DetectionTransform):
    """Flip the image and boxes horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.transform_op = T.RandomHorizontalFlip(p=1.0)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        return self._apply_torchvision_transform(input_dict, self.transform_op)


class RandomIoUCrop(DetectionTransform):
    """Sample a crop with minimum-IoU constraints."""

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        p: float = 1.0,
        stop_epoch: int | None = None,
    ) -> None:
        super().__init__(p=p, stop_epoch=stop_epoch)
        self.transform_op = T.RandomIoUCrop(
            min_scale=min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=sampler_options,
            trials=trials,
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        return self._apply_torchvision_transform(input_dict, self.transform_op)


class Resize(DetectionTransform):
    """Resize images and boxes to a fixed size."""

    def __init__(self, size: int | tuple[int, int], max_size: int | None = None) -> None:
        super().__init__(p=1.0)
        self.transform_op = T.Resize(size=size, max_size=max_size)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        return self._apply_torchvision_transform(input_dict, self.transform_op)


class SanitizeBoundingBoxes(DetectionTransform):
    """Remove invalid boxes and keep target fields aligned."""

    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__(p=1.0)
        self.min_size = min_size

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        target = clone_target(input_dict["target"])
        boxes = target["boxes"]
        has_box_metadata = hasattr(boxes, "canvas_size")
        box_canvas_size = tuple(boxes.canvas_size) if has_box_metadata else None
        box_format = boxes.format.value if has_box_metadata else "XYXY"
        boxes_tensor = boxes.as_subclass(torch.Tensor) if has_box_metadata else boxes
        if boxes_tensor.numel() == 0:
            empty_boxes = boxes_tensor.reshape(0, 4)
            if has_box_metadata:
                empty_boxes = convert_boxes_to_tv_tensor(
                    empty_boxes,
                    canvas_size=box_canvas_size,
                    box_format=box_format,
                )
            target["boxes"] = empty_boxes
            target["labels"] = target["labels"][:0]
            target["area"] = boxes_tensor.new_zeros((0,))
            if "iscrowd" in target:
                target["iscrowd"] = target["iscrowd"][:0]
            return {"target": target}

        widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
        heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
        keep = (widths >= self.min_size) & (heights >= self.min_size)

        kept_boxes = boxes_tensor[keep]
        if has_box_metadata:
            target["boxes"] = convert_boxes_to_tv_tensor(
                kept_boxes,
                canvas_size=box_canvas_size,
                box_format=box_format,
            )
        else:
            target["boxes"] = kept_boxes
        target["labels"] = target["labels"][keep]
        if "iscrowd" in target:
            target["iscrowd"] = target["iscrowd"][keep]
        target["area"] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
        return {"target": target}


class Mosaic(DetectionTransform):
    """Cache-based mosaic augmentation adapted from RT-DETR."""

    def __init__(
        self,
        output_size: int = 320,
        max_size: int | None = None,
        rotation_range: float = 0.0,
        translation_range: tuple[float, float] = (0.1, 0.1),
        scaling_range: tuple[float, float] = (0.5, 1.5),
        probability: float = 1.0,
        fill_value: int = 114,
        max_cached_images: int = 50,
        random_pop: bool = True,
        stop_epoch: int | None = None,
    ) -> None:
        super().__init__(p=probability, stop_epoch=stop_epoch)
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.affine_transform = T.RandomAffine(
            degrees=rotation_range,
            translate=translation_range,
            scale=scaling_range,
            fill=fill_value,
        )
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.cache: list[dict[str, Any]] = []

    def _clone_sample(self, image: Image.Image, target: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": image.copy(),
            "target": clone_target(target),
        }

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        image = input_dict["image"]
        target = clone_target(input_dict["target"])
        image, target = self.resize(image, target)
        self.cache.append(self._clone_sample(image, target))
        if len(self.cache) > self.max_cached_images:
            pop_index = random.randint(0, len(self.cache) - 2) if self.random_pop else 0
            self.cache.pop(pop_index)

        if len(self.cache) < 4:
            return {"image": image, "target": target}

        sample_indices = random.choices(range(len(self.cache)), k=3)
        samples = [self._clone_sample(image, target)] + [
            self._clone_sample(self.cache[idx]["image"], self.cache[idx]["target"]) for idx in sample_indices
        ]

        sizes = [(sample["image"].height, sample["image"].width) for sample in samples]
        max_height = max(height for height, _ in sizes)
        max_width = max(width for _, width in sizes)

        placement_offsets = [(0, 0), (max_width, 0), (0, max_height), (max_width, max_height)]
        merged_image = Image.new(mode=samples[0]["image"].mode, size=(max_width * 2, max_height * 2), color=0)

        merged_labels: list[torch.Tensor] = []
        merged_boxes: list[torch.Tensor] = []
        merged_iscrowd: list[torch.Tensor] = []

        for sample, (offset_x, offset_y) in zip(samples, placement_offsets, strict=False):
            merged_image.paste(sample["image"], (offset_x, offset_y))
            sample_boxes = sample["target"]["boxes"].as_subclass(torch.Tensor).clone()
            sample_boxes[:, [0, 2]] += offset_x
            sample_boxes[:, [1, 3]] += offset_y
            merged_boxes.append(sample_boxes)
            merged_labels.append(sample["target"]["labels"])
            if "iscrowd" in sample["target"]:
                merged_iscrowd.append(sample["target"]["iscrowd"])

        mosaic_target = clone_target(target)
        mosaic_target["boxes"] = convert_boxes_to_tv_tensor(
            torch.cat(merged_boxes, dim=0),
            canvas_size=(max_height * 2, max_width * 2),
            box_format="XYXY",
        )
        mosaic_target["labels"] = torch.cat(merged_labels, dim=0)
        if merged_iscrowd:
            mosaic_target["iscrowd"] = torch.cat(merged_iscrowd, dim=0)

        merged_image, mosaic_target = self.affine_transform(merged_image, mosaic_target)
        return {"image": merged_image, "target": mosaic_target}
