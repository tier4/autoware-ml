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

"""T4Dataset datamodule for combined PTv3 segmentation+detection evaluation."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from typing import Any

from torch.utils.data import DataLoader

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.detection3d import (
    build_detection_dataloader,
    build_label_to_category,
    load_detection_data_infos,
    normalize_detection_sample,
    resolve_data_path,
    resolve_sweep_paths,
)
from autoware_ml.datamodule.t4dataset.detection3d import (
    FrameSamplingConfig,
    coerce_frame_sampling,
    compute_frame_sampling_weights,
)
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.annotations import normalize_filter_attributes


class T4SegmentationDetection3DDataset(Dataset):
    """T4 dataset for combined PTv3 segmentation+detection from a single annotation file."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        use_valid_flag: bool = False,
        frame_sampling: FrameSamplingConfig | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the combined T4 segmentation+detection dataset."""
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.use_valid_flag = use_valid_flag
        self.frame_sampling = frame_sampling

        with open(ann_file, "rb") as file:
            data = pickle.load(file)
        self.label_to_category = build_label_to_category(data.get("metainfo", {}))

        raw_infos = load_detection_data_infos(data)

        self.data_infos: list[dict[str, Any]] = []
        for raw_sample in raw_infos:
            if "pts_semantic_mask_path" not in raw_sample:
                raise ValueError(
                    f"Record with token '{raw_sample.get('token', '<unknown>')}' is missing "
                    f"'pts_semantic_mask_path'. The combined segdet annotation file must contain "
                    f"both detection and segmentation fields."
                )
            sample = normalize_detection_sample(raw_sample)
            sample["label_to_category"] = self.label_to_category
            sample["pts_semantic_mask_path"] = raw_sample["pts_semantic_mask_path"]
            sample["pts_semantic_mask_categories"] = raw_sample["pts_semantic_mask_categories"]
            self.data_infos.append(sample)

        self.frame_weights = compute_frame_sampling_weights(
            self.data_infos,
            self.class_names,
            self.name_mapping,
            self.frame_sampling,
            self.filter_attributes,
            self.use_valid_flag,
        )

    def __len__(self) -> int:
        """Return the number of T4 segdet samples."""
        return len(self.data_infos)

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one combined metadata record consumed by the transform pipeline."""
        sample = self.data_infos[index]
        return {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "name_mapping": self.name_mapping,
            "label_to_category": self.label_to_category,
            "sample_token": sample["token"],
            "lidar_path": resolve_data_path(self.data_root, sample["lidar_path"]),
            "num_pts_feats": int(sample["lidar_points"].get("num_pts_feats", 5)),
            "sweeps": resolve_sweep_paths(sample, self.data_root),
            "pts_semantic_mask_categories": sample["pts_semantic_mask_categories"],
            "pts_semantic_mask_path": resolve_data_path(
                self.data_root, sample["pts_semantic_mask_path"]
            ),
        }


class T4SegmentationDetection3DDataModule(DataModule):
    """Create T4 dataloaders for combined PTv3 segmentation+detection evaluation."""

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        use_valid_flag: bool = False,
        train_frame_sampling: FrameSamplingConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the combined T4 segmentation+detection datamodule."""
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.use_valid_flag = use_valid_flag
        self.train_frame_sampling = coerce_frame_sampling(train_frame_sampling)

        def resolve_ann_file(ann_file: str) -> str:
            return ann_file if os.path.isabs(ann_file) else os.path.join(data_root, ann_file)

        self.ann_files = {
            "train": resolve_ann_file(train_ann_file),
            "val": resolve_ann_file(val_ann_file),
            "test": resolve_ann_file(test_ann_file),
            "predict": resolve_ann_file(test_ann_file),
        }

    def _create_dataset(
        self, split: str, dataset_transforms: TransformsCompose | None = None
    ) -> Dataset:
        """Instantiate the combined dataset for one split."""
        return T4SegmentationDetection3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            class_names=self.class_names,
            name_mapping=self.name_mapping,
            filter_attributes=self.filter_attributes,
            use_valid_flag=self.use_valid_flag,
            frame_sampling=self.train_frame_sampling if split == "train" else None,
            dataset_transforms=dataset_transforms,
        )

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create a joint dataloader with optional train RFS sampling."""
        return build_detection_dataloader(
            dataset=getattr(self, f"{split}_dataset"),
            dataloader_cfg=getattr(self, f"{split}_dataloader_cfg"),
            is_train=split == "train",
            train_frame_sampling=self.train_frame_sampling,
            collate_fn=self.collate_fn,
        )
