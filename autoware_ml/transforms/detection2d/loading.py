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

"""Loading and formatting transforms for detection2d pipelines."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageFile
import torch
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as F

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.detection2d.utils import convert_boxes_to_tv_tensor, clone_target

# Avoid DecompressionBomb warnings/errors on trusted local training data.
Image.MAX_IMAGE_PIXELS = None

# Allow JPEG files that PIL reports as truncated even though they can be decoded and used for training.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class LoadDetectionImageFromFile(BaseTransform):
    """Load one RGB image for detection tasks using PIL."""

    _required_keys = ["img_path", "target"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(input_dict["img_path"]).convert("RGB")
        return {"image": image, "target": clone_target(input_dict["target"])}


class ToTorchVisionTensors(BaseTransform):
    """Convert plain detection target tensors into torchvision tv_tensors."""

    _required_keys = ["image", "target"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        image = input_dict["image"]
        target = clone_target(input_dict["target"])
        boxes = target["boxes"]
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(-1, 4)
        target["boxes"] = convert_boxes_to_tv_tensor(
            boxes.float(),
            canvas_size=(image.height, image.width),
            box_format="XYXY",
        )
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        if "iscrowd" in target:
            target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
        if "area" in target:
            target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)
        return {"target": target}


class ConvertPILImage(BaseTransform):
    """Convert a PIL image into a float tensor."""

    _required_keys = ["image"]

    def __init__(self, dtype: str = "float32", scale: bool = True) -> None:
        self.dtype = dtype
        self.scale = scale

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        image = F.pil_to_tensor(input_dict["image"])
        if self.dtype == "float32":
            image = image.float()
        if self.scale:
            image = image / 255.0
        return {"image": image}


class ConvertBoxes(BaseTransform):
    """Convert target boxes into a plain tensor format expected by the model."""

    _required_keys = ["target"]

    def __init__(self, fmt: str = "cxcywh", normalize: bool = False) -> None:
        self.fmt = fmt.upper()
        self.normalize = normalize

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        target = clone_target(input_dict["target"])
        boxes = target["boxes"]
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if hasattr(boxes, "format"):
            source_format = boxes.format.value.lower()
            canvas_size = tuple(int(v) for v in boxes.canvas_size)
        else:
            source_format = "xyxy"
            canvas_size = None
        if canvas_size is None and "image" in input_dict:
            image = input_dict["image"]
            if hasattr(image, "shape"):
                canvas_size = (int(image.shape[-2]), int(image.shape[-1]))
            elif hasattr(image, "height") and hasattr(image, "width"):
                canvas_size = (int(image.height), int(image.width))
        if self.fmt:
            boxes = box_convert(boxes, in_fmt=source_format, out_fmt=self.fmt.lower())
        boxes = boxes.float()
        if self.normalize:
            if canvas_size is None:
                raise ValueError("Cannot normalize boxes without a canvas size.")
            normalizer = torch.tensor(
                [canvas_size[1], canvas_size[0], canvas_size[1], canvas_size[0]],
                dtype=boxes.dtype,
                device=boxes.device,
            )
            boxes = boxes / normalizer
        target["boxes"] = boxes
        if boxes.numel() == 0:
            target["area"] = boxes.new_zeros((0,))
        elif self.fmt == "CXCYWH":
            target["area"] = boxes[:, 2] * boxes[:, 3]
        else:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return {"target": target}
