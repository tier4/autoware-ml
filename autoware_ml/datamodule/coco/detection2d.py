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

"""COCO-style detection datamodule implementations."""

from __future__ import annotations

import os
from typing import Any

from autoware_ml.datamodule.detection2d.base import CocoStyleDetectionDataset, DetectionDataModule
from autoware_ml.transforms.base import TransformsCompose


class COCODetectionDataModule(DetectionDataModule):
    """DataModule for COCO-style 2D detection datasets."""

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        train_img_root: str = "",
        val_img_root: str = "",
        test_ann_file: str | None = None,
        test_img_root: str | None = None,
        filter_empty_gt: bool = False,
        min_img_size: int | None = None,
        train_sample_ids: list[int] | None = None,
        val_sample_ids: list[int] | None = None,
        test_sample_ids: list[int] | None = None,
        predict_sample_ids: list[int] | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        max_predict_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_root = data_root
        self.ann_files = {
            "train": train_ann_file,
            "val": val_ann_file,
            "test": test_ann_file or val_ann_file,
            "predict": test_ann_file or val_ann_file,
        }
        self.img_roots = {
            "train": train_img_root,
            "val": val_img_root,
            "test": test_img_root or val_img_root,
            "predict": test_img_root or val_img_root,
        }
        self.filter_empty_gt = filter_empty_gt
        self.min_img_size = min_img_size
        self.sample_ids = {
            "train": train_sample_ids,
            "val": val_sample_ids,
            "test": test_sample_ids,
            "predict": predict_sample_ids,
        }
        self.max_samples = {
            "train": max_train_samples,
            "val": max_val_samples,
            "test": max_test_samples,
            "predict": max_predict_samples,
        }

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def _create_dataset(
        self,
        split: str,
        transforms: TransformsCompose | None = None,
    ) -> CocoStyleDetectionDataset:
        return CocoStyleDetectionDataset(
            ann_file=self._resolve_path(self.ann_files[split]),
            data_root=self.data_root,
            img_root=self._resolve_path(self.img_roots[split]) if self.img_roots[split] else self.data_root,
            dataset_transforms=transforms,
            filter_empty_gt=self.filter_empty_gt and split == "train",
            min_img_size=self.min_img_size,
            sample_ids=self.sample_ids[split],
            max_samples=self.max_samples[split],
        )
