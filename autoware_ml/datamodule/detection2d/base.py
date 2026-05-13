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

"""Reusable datamodule primitives for COCO-style 2D detection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
import random
from typing import Any

import torch
import torch.nn.functional as F

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.metrics.detection2d import build_coco_api_from_dataset_dict
from autoware_ml.transforms.base import TransformsCompose


def generate_scales(base_size: int, base_size_repeat: int) -> list[int]:
    """Generate RT-DETR-style multiscale image sizes."""
    min_size = int(base_size * 0.75 / 32) * 32
    scale_repeat = (base_size - min_size) // 32
    scales = [min_size + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


class CocoStyleDetectionDataset(Dataset):
    """Dataset for COCO-style JSON annotations with contiguous training labels."""

    def __init__(
        self,
        ann_file: str,
        data_root: str = "",
        img_root: str = "",
        dataset_transforms: TransformsCompose | None = None,
        filter_empty_gt: bool = False,
        min_img_size: int | None = None,
        sample_ids: Sequence[int] | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(dataset_transforms=dataset_transforms)
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_root = img_root
        self.filter_empty_gt = filter_empty_gt
        self.min_img_size = min_img_size
        self.sample_ids = None if sample_ids is None else {int(sample_id) for sample_id in sample_ids}
        self.max_samples = max_samples
        self.owner_datamodule: Any | None = None

        with open(ann_file, "r", encoding="utf-8") as file:
            self.dataset_dict = json.load(file)

        self.categories = list(self.dataset_dict.get("categories", []))
        self.category_id_to_label = {
            int(category["id"]): index for index, category in enumerate(self.categories)
        }
        self.label_to_category_id = {
            label: category_id for category_id, label in self.category_id_to_label.items()
        }

        self.images = list(self.dataset_dict.get("images", []))
        annotation_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.dataset_dict.get("annotations", []):
            annotation_map[int(annotation["image_id"])].append(annotation)
        self.annotation_map = dict(annotation_map)

        self.indices = []
        for index, image_info in enumerate(self.images):
            image_id = int(image_info["id"])
            if self.sample_ids is not None and image_id not in self.sample_ids:
                continue
            annotations = self.annotation_map.get(int(image_info["id"]), [])
            if self.filter_empty_gt and not annotations:
                continue
            if self.min_img_size is not None and (
                int(image_info.get("width", 0)) < self.min_img_size
                or int(image_info.get("height", 0)) < self.min_img_size
            ):
                continue
            self.indices.append(index)
            if self.max_samples is not None and len(self.indices) >= self.max_samples:
                break

        self.selected_image_ids = {int(self.images[index]["id"]) for index in self.indices}
        self.filtered_dataset_dict = {
            **self.dataset_dict,
            "images": [self.images[index] for index in self.indices],
            "annotations": [
                annotation
                for annotation in self.dataset_dict.get("annotations", [])
                if int(annotation["image_id"]) in self.selected_image_ids
            ],
            "categories": self.categories,
        }

        self._coco_api = None

    def __len__(self) -> int:
        return len(self.indices)

    def _resolve_img_path(self, file_name: str) -> str:
        if os.path.isabs(file_name):
            return file_name
        base_dir = self.img_root or self.data_root
        return os.path.join(base_dir, file_name)

    def get_coco_api(self) -> Any:
        if self._coco_api is None:
            self._coco_api = build_coco_api_from_dataset_dict(self.filtered_dataset_dict)
        return self._coco_api

    def get_data_info(self, index: int) -> dict[str, Any]:
        image_info = self.images[self.indices[index]]
        image_id = int(image_info["id"])
        width = int(image_info["width"])
        height = int(image_info["height"])

        boxes = []
        labels = []
        iscrowd = []
        areas = []
        for annotation in self.annotation_map.get(image_id, []):
            x, y, w, h = annotation["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[int(annotation["category_id"])])
            iscrowd.append(int(annotation.get("iscrowd", 0)))
            areas.append(float(annotation.get("area", w * h)))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        area_tensor = torch.tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "iscrowd": iscrowd_tensor,
            "area": area_tensor,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }

        return {
            "img_path": self._resolve_img_path(image_info["file_name"]),
            "img_id": image_id,
            "orig_size": target["orig_size"],
            "target": target,
            "metadata": image_info,
        }


class DetectionDataModule(DataModule):
    """Base datamodule with detection-specific collation and augmentation scheduling."""

    def __init__(
        self,
        multiscale_sizes: Sequence[int] | None = None,
        multiscale_base_size: int | None = None,
        multiscale_base_size_repeat: int = 3,
        multiscale_stop_epoch: int | None = None,
        mixup_prob: float = 0.0,
        mixup_epochs: Sequence[int] = (0, 0),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if multiscale_sizes is not None:
            self.multiscale_sizes = list(multiscale_sizes)
        elif multiscale_base_size is not None:
            self.multiscale_sizes = generate_scales(multiscale_base_size, multiscale_base_size_repeat)
        else:
            self.multiscale_sizes = None
        self.multiscale_stop_epoch = multiscale_stop_epoch
        self.mixup_prob = mixup_prob
        self.mixup_epochs = list(mixup_epochs)

    @property
    def current_epoch(self) -> int:
        trainer = getattr(self, "trainer", None)
        return int(trainer.current_epoch) if trainer is not None else 0

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage=stage)
        for split in ("train", "val", "test", "predict"):
            dataset = getattr(self, f"{split}_dataset")
            if dataset is not None:
                dataset.owner_datamodule = self

    def _apply_mixup(
        self,
        images: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        trainer = getattr(self, "trainer", None)
        if trainer is None or not trainer.training:
            return images, targets
        if self.mixup_prob <= 0.0 or len(self.mixup_epochs) < 2:
            return images, targets
        if not (self.mixup_epochs[0] <= self.current_epoch < self.mixup_epochs[-1]):
            return images, targets
        if random.random() >= self.mixup_prob or len(targets) < 2:
            return images, targets

        beta = round(random.uniform(0.45, 0.55), 6)
        mixed_images = images.roll(shifts=1, dims=0).mul(1.0 - beta).add(images.mul(beta))

        shifted_targets = targets[-1:] + targets[:-1]
        updated_targets: list[dict[str, Any]] = []
        for target, shifted_target in zip(targets, shifted_targets, strict=False):
            merged = {key: value for key, value in target.items()}
            merged["boxes"] = torch.cat([target["boxes"], shifted_target["boxes"]], dim=0)
            merged["labels"] = torch.cat([target["labels"], shifted_target["labels"]], dim=0)
            if "iscrowd" in target and "iscrowd" in shifted_target:
                merged["iscrowd"] = torch.cat([target["iscrowd"], shifted_target["iscrowd"]], dim=0)
            merged["mixup"] = torch.tensor(
                [beta] * len(target["labels"]) + [1.0 - beta] * len(shifted_target["labels"]),
                dtype=torch.float32,
            )
            updated_targets.append(merged)

        return mixed_images, updated_targets

    def _apply_multiscale(self, images: torch.Tensor) -> torch.Tensor:
        trainer = getattr(self, "trainer", None)
        if trainer is None or not trainer.training:
            return images
        if not self.multiscale_sizes:
            return images
        if self.multiscale_stop_epoch is not None and self.current_epoch >= self.multiscale_stop_epoch:
            return images
        size = random.choice(self.multiscale_sizes)
        return F.interpolate(images, size=size, mode="bilinear", align_corners=False)

    def collate_fn(self, batch_inputs_dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not batch_inputs_dicts:
            raise ValueError("Batch inputs dictionary is empty.")

        images = torch.stack([sample["image"] for sample in batch_inputs_dicts], dim=0)
        targets = [sample["target"] for sample in batch_inputs_dicts]
        images, targets = self._apply_mixup(images, targets)
        images = self._apply_multiscale(images)

        return {
            "images": images,
            "targets": targets,
            "image_ids": torch.tensor([sample["img_id"] for sample in batch_inputs_dicts], dtype=torch.int64),
            "orig_sizes": torch.stack([sample["orig_size"] for sample in batch_inputs_dicts], dim=0),
            "img_paths": [sample["img_path"] for sample in batch_inputs_dicts],
            "metadata": [sample.get("metadata") for sample in batch_inputs_dicts],
        }
