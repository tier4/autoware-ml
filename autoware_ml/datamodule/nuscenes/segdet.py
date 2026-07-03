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

"""NuScenes datamodule for combined PTv3 segmentation+detection evaluation."""

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
)
from autoware_ml.datamodule.nuscenes.common import resolve_lidar_path
from autoware_ml.datamodule.t4dataset.detection3d import (
    FrameSamplingConfig,
    coerce_frame_sampling,
    compute_frame_sampling_weights,
)
from autoware_ml.transforms.base import TransformsCompose


def _resolve_path(base: str, path: str) -> str:
    """Join *path* to *base* unless *path* is already absolute."""
    return path if os.path.isabs(path) else os.path.join(base, path)


class NuscenesSegmentationDetection3DDataset(Dataset):
    """NuScenes dataset for combined PTv3 segmentation+detection from a unified info file.

    The unified NuScenes info carries both detection ``instances`` and a
    ``pts_semantic_mask_path`` per sample. Unlike the T4 combined dataset,
    NuScenes annotations do not provide ``pts_semantic_mask_categories``; point
    labels are remapped downstream via a raw-label mapping in the transform
    pipeline instead of a per-sample category table.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str] | None = None,
        lidarseg_dir: str = "lidarseg/v1.0-trainval",
        frame_sampling: FrameSamplingConfig | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the combined NuScenes segmentation+detection dataset.

        Args:
            data_root: Dataset root directory.
            ann_file: Unified annotation file path.
            class_names: Ordered detector class names.
            name_mapping: Optional raw-label to detector-label mapping.
            lidarseg_dir: Directory containing lidarseg label files. Joined to
                *data_root* when relative.
            frame_sampling: Optional repeat-factor sampling configuration.
            dataset_transforms: Optional dataset transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        self.lidarseg_dir = _resolve_path(data_root, lidarseg_dir)
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
            self.data_infos.append(sample)

        self.frame_weights = compute_frame_sampling_weights(
            self.data_infos,
            self.class_names,
            self.name_mapping or {},
            self.frame_sampling,
        )

    def __len__(self) -> int:
        """Return the number of NuScenes segdet samples."""
        return len(self.data_infos)

    def _resolve_sweeps(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Resolve sweep lidar paths for one sample using NuScenes conventions."""
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
        """Build one combined metadata record consumed by the transform pipeline."""
        sample = self.data_infos[index]
        return {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "name_mapping": self.name_mapping,
            "label_to_category": self.label_to_category,
            "sample_token": sample["token"],
            "lidar_path": resolve_lidar_path(self.data_root, sample["lidar_path"]),
            "num_pts_feats": int(
                sample.get("num_features", sample.get("lidar_points", {}).get("num_pts_feats", 5))
            ),
            "sweeps": self._resolve_sweeps(sample),
            "pts_semantic_mask_path": os.path.join(
                self.lidarseg_dir, sample["pts_semantic_mask_path"]
            ),
        }


class NuscenesSegmentationDetection3DDataModule(DataModule):
    """Create NuScenes dataloaders for combined PTv3 segmentation+detection evaluation."""

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        class_names: list[str],
        lidarseg_dir: str = "lidarseg/v1.0-trainval",
        name_mapping: Mapping[str, str] | None = None,
        train_frame_sampling: FrameSamplingConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the combined NuScenes segmentation+detection datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            class_names: Ordered detector class names.
            lidarseg_dir: Directory containing lidarseg label files.
            name_mapping: Optional raw-label to detector-label mapping.
            train_frame_sampling: Optional repeat-factor sampling for training.
            **kwargs: Additional base datamodule configuration.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.lidarseg_dir = lidarseg_dir
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        self.train_frame_sampling = coerce_frame_sampling(train_frame_sampling)

        self.ann_files = {
            "train": _resolve_path(data_root, train_ann_file),
            "val": _resolve_path(data_root, val_ann_file),
            "test": _resolve_path(data_root, test_ann_file),
            "predict": _resolve_path(data_root, test_ann_file),
        }

    def _create_dataset(
        self, split: str, dataset_transforms: TransformsCompose | None = None
    ) -> Dataset:
        """Instantiate the combined dataset for one split."""
        return NuscenesSegmentationDetection3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            class_names=self.class_names,
            name_mapping=self.name_mapping,
            lidarseg_dir=self.lidarseg_dir,
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
