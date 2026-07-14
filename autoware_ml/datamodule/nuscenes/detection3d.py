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

"""NuScenes 3D detection dataset and datamodule.

This module adapts NuScenes lidar detection annotations to the shared
Autoware-ML detection datamodule interface.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import pickle
from typing import Any

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.detection3d import (
    build_label_to_category,
    load_detection_data_infos,
)
from autoware_ml.datamodule.nuscenes.common import resolve_lidar_path
from autoware_ml.transforms.base import TransformsCompose


class NuscenesDetection3DDataset(Dataset):
    """Load NuScenes lidar samples for 3D object detection.

    The dataset returns metadata consumed by transform pipelines that load
    lidar points and detection annotations on demand.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str] | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the NuScenes detection dataset.

        Args:
            data_root: Dataset root directory.
            ann_file: Annotation file path.
            class_names: Ordered detector class names.
            name_mapping: Optional raw-label to detector-label mapping.
            dataset_transforms: Optional dataset transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        with open(ann_file, "rb") as file:
            data = pickle.load(file)
        self.data_infos = load_detection_data_infos(data)
        self.label_to_category = build_label_to_category(data.get("metainfo", {}))

    def __len__(self) -> int:
        """Return the number of annotated samples.

        Returns:
            Number of samples available in the annotation file.
        """
        return len(self.data_infos)

    def _resolve_lidar_path(self, sample: Mapping[str, Any]) -> str:
        """Resolve the lidar path for one sample.

        Args:
            sample: Annotation entry for one frame.

        Returns:
            Absolute lidar path for the sample.
        """
        return resolve_lidar_path(self.data_root, sample["lidar_path"])

    def _resolve_sweeps(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Resolve sweep lidar paths for one sample.

        Args:
            sample: Annotation entry for one frame.

        Returns:
            Sweep metadata list with absolute ``lidar_path`` values.
        """
        sweep_entries = []
        for sweep in sample.get("sweeps", []):
            sweep_entry = dict(sweep)
            if "lidar_path" in sweep_entry:
                sweep_entry["lidar_path"] = resolve_lidar_path(
                    self.data_root, sweep_entry["lidar_path"]
                )
            sweep_entries.append(sweep_entry)
        return sweep_entries

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one NuScenes detection metadata record.

        Args:
            index: Dataset sample index.

        Returns:
            Metadata dictionary consumed by detection transform pipelines.
        """
        sample = self.data_infos[index]
        lidar_path = self._resolve_lidar_path(sample)
        return {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "name_mapping": self.name_mapping,
            "label_to_category": self.label_to_category,
            "sample_token": sample["token"],
            "timestamp": sample.get("timestamp"),
            "lidar_path": lidar_path,
            "num_pts_feats": int(
                sample.get("num_features", sample.get("lidar_points", {}).get("num_pts_feats", 5))
            ),
            "sweeps": self._resolve_sweeps(sample),
        }


class NuscenesDetection3DDataModule(DataModule):
    """Create NuScenes dataloaders for 3D object detection.

    The datamodule wires split-specific datasets, transforms, and dataloader
    settings for lidar-based detection models.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str] | None = None,
        train_frame_sampling: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the NuScenes detection datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            class_names: Ordered detector class names.
            name_mapping: Optional raw-label to detector-label mapping.
            train_frame_sampling: Unsupported for NuScenes detection.
            **kwargs: Additional base datamodule configuration.
        """
        if train_frame_sampling is not None:
            raise ValueError("NuscenesDetection3DDataModule does not support train_frame_sampling.")
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None

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
        """Instantiate the dataset for one split.

        Args:
            split: Dataset split name.
            dataset_transforms: Optional transform pipeline for the split.

        Returns:
            Instantiated dataset for the requested split.
        """
        return NuscenesDetection3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            class_names=self.class_names,
            name_mapping=self.name_mapping,
            dataset_transforms=dataset_transforms,
        )
