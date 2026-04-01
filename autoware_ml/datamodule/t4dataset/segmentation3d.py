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

"""T4Dataset 3D semantic segmentation dataset and datamodule.

This module adapts T4Dataset segmentation metadata to the shared
Autoware-ML segmentation interface.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.transforms.base import TransformsCompose

logger = logging.getLogger(__name__)


def _resolve_path(base: str, path: str) -> str:
    """Join *path* to *base* unless *path* is already absolute."""
    return path if os.path.isabs(path) else os.path.join(base, path)


class T4Segmentation3DDataset(Dataset):
    """Load T4Dataset lidar samples for point-wise semantic segmentation.

    When *lidar_sources* is provided the dataset flattens per-sensor metadata
    so that each ``(sample, sensor)`` pair becomes a separate item.  When
    omitted, each annotation record maps 1-to-1 to a dataset item.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        lidar_sources: list[str] | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the T4 segmentation dataset.

        Args:
            data_root: Dataset root directory.
            ann_file: Annotation file path.
            lidar_sources: Ordered lidar sources exposed as separate samples.
                When ``None`` each annotation record is one sample.
            dataset_transforms: Optional dataset transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.lidar_sources = lidar_sources
        with open(ann_file, "rb") as file:
            data = pickle.load(file)

        self.data_infos = data["data_list"] if "data_list" in data else data["infos"]

    def __len__(self) -> int:
        """Return the number of segmentation samples.

        Returns:
            Total number of items.  When *lidar_sources* is set this equals
            ``len(data_infos) * len(lidar_sources)``.
        """
        if self.lidar_sources is not None:
            return len(self.data_infos) * len(self.lidar_sources)
        return len(self.data_infos)

    def _map_index(self, index: int) -> tuple[int, str]:
        """Map a flattened dataset index to sample and lidar source.

        Args:
            index: Flattened dataset index.

        Returns:
            Tuple of sample index and lidar source name.
        """
        assert self.lidar_sources is not None
        source_count = len(self.lidar_sources)
        sample_index = index // source_count
        source_name = self.lidar_sources[index % source_count]
        return sample_index, source_name

    def _get_per_sensor_info(
        self, sample: dict[str, Any], sample_index: int, source_name: str
    ) -> dict[str, Any]:
        """Build metadata for one sensor of a multi-sensor sample."""
        lidar_path = os.path.join(self.data_root, sample["lidar_points"]["lidar_path"])
        sensor_token = sample["lidar_sources"][source_name]["sensor_token"]
        sources_by_token = {s["sensor_token"]: s for s in sample["lidar_sources_info"]["sources"]}
        source_info = sources_by_token[sensor_token]
        idx_begin, length = source_info["idx_begin"], source_info["length"]
        translation = sample["lidar_sources"][source_name]["translation"]
        rotation = sample["lidar_sources"][source_name]["rotation"]

        if length <= 0:
            logger.warning(
                "Sample %s source %r has no points (length=%s).",
                sample_index,
                source_name,
                length,
            )

        return {
            "lidar_path": lidar_path,
            "name": sample.get("token", f"{sample_index}_{source_name}"),
            "idx_begin": idx_begin,
            "length": length,
            "translation": translation,
            "rotation": rotation,
            "num_pts_feats": sample["lidar_points"]["num_pts_feats"],
            "pts_semantic_mask_categories": sample["pts_semantic_mask_categories"],
            "pts_semantic_mask_path": os.path.join(
                self.data_root, sample["pts_semantic_mask_path"]
            ),
        }

    def _get_sample_info(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Build metadata for a single-sensor sample (no per-sensor flattening)."""
        lidar_path = os.path.join(self.data_root, sample["lidar_points"]["lidar_path"])
        return {
            "lidar_path": lidar_path,
            "name": sample["token"],
            "num_pts_feats": int(sample["lidar_points"].get("num_pts_feats", 5)),
            "pts_semantic_mask_categories": sample["pts_semantic_mask_categories"],
            "pts_semantic_mask_path": os.path.join(
                self.data_root, sample["pts_semantic_mask_path"]
            ),
        }

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one T4 segmentation metadata record.

        Args:
            index: Dataset index (flattened when *lidar_sources* is set).

        Returns:
            Metadata dictionary consumed by segmentation transform pipelines.
        """
        if self.lidar_sources is not None:
            sample_index, source_name = self._map_index(index)
            sample = self.data_infos[sample_index]
            return self._get_per_sensor_info(sample, sample_index, source_name)

        sample = self.data_infos[index]
        return self._get_sample_info(sample)


class T4Segmentation3DDataModule(DataModule):
    """Create T4Dataset dataloaders for 3D semantic segmentation.

    The datamodule configures dataset splits, transforms, and collate behavior
    for segmentation experiments on T4Dataset.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        lidar_sources: list[str] | None = None,
        mix_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the T4 segmentation datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            lidar_sources: Ordered lidar sources exposed as separate samples.
                When ``None`` each annotation record is one sample.
            mix_prob: Mix probability forwarded to the point-cloud collate
                function during training.  Set to ``0.0`` to disable.
            **kwargs: Additional base datamodule configuration.
        """
        super().__init__(mix_prob=mix_prob, **kwargs)
        self.data_root = data_root
        self.lidar_sources = lidar_sources
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
        return T4Segmentation3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            lidar_sources=self.lidar_sources,
            dataset_transforms=dataset_transforms,
        )
