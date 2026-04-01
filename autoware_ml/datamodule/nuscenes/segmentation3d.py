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

"""NuScenes 3D semantic segmentation dataset and datamodule.

This module adapts NuScenes lidar segmentation metadata to the shared
point-cloud segmentation datamodule interface.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.nuscenes.common import resolve_lidar_path
from autoware_ml.transforms.base import TransformsCompose


def _resolve_path(base: str, path: str) -> str:
    """Join *path* to *base* unless *path* is already absolute."""
    return path if os.path.isabs(path) else os.path.join(base, path)


class NuscenesSegmentation3DDataset(Dataset):
    """Load NuScenes lidar samples for point-wise semantic segmentation.

    The dataset returns sample metadata consumed by transform pipelines that
    load lidar points and semantic masks on demand.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        lidarseg_dir: str = "lidarseg/v1.0-trainval",
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the NuScenes segmentation dataset.

        Args:
            data_root: Dataset root directory.
            ann_file: Annotation file path.
            lidarseg_dir: Directory containing lidarseg label files.  Joined to
                *data_root* when relative.
            dataset_transforms: Optional dataset transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.lidarseg_dir = _resolve_path(data_root, lidarseg_dir)
        with open(ann_file, "rb") as file:
            data = pickle.load(file)

        self.data_infos = data["data_list"] if "data_list" in data else data["infos"]

    def __len__(self) -> int:
        """Return the number of annotated samples.

        Returns:
            Number of samples available in the annotation file.
        """
        return len(self.data_infos)

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one NuScenes segmentation metadata record.

        Args:
            index: Dataset sample index.

        Returns:
            Metadata dictionary consumed by segmentation transform pipelines.
        """
        sample = self.data_infos[index]
        lidar_path = resolve_lidar_path(self.data_root, sample["lidar_points"]["lidar_path"])

        return {
            "lidar_path": lidar_path,
            "name": sample["token"],
            "num_pts_feats": int(sample.get("lidar_points", {}).get("num_pts_feats", 5)),
            "pts_semantic_mask_path": os.path.join(
                self.lidarseg_dir, sample["pts_semantic_mask_path"]
            ),
        }


class NuscenesSegmentation3DDataModule(DataModule):
    """Create NuScenes dataloaders for 3D semantic segmentation.

    The datamodule assembles segmentation datasets and collate behavior for
    segmentation model training, evaluation, and export.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        lidarseg_dir: str = "lidarseg/v1.0-trainval",
        mix_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the NuScenes segmentation datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            lidarseg_dir: Directory containing lidarseg label files.
            mix_prob: Mix probability forwarded to the point-cloud collate
                function during training.  Set to ``0.0`` to disable.
            **kwargs: Additional base datamodule configuration.
        """
        super().__init__(mix_prob=mix_prob, **kwargs)
        self.data_root = data_root
        self.lidarseg_dir = lidarseg_dir

        self.ann_files = {
            "train": _resolve_path(data_root, train_ann_file),
            "val": _resolve_path(data_root, val_ann_file),
            "test": _resolve_path(data_root, test_ann_file),
            "predict": _resolve_path(data_root, test_ann_file),
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
        return NuscenesSegmentation3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            lidarseg_dir=self.lidarseg_dir,
            dataset_transforms=dataset_transforms,
        )
