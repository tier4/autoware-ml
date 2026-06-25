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

"""T4Dataset 3D detection dataset and datamodule.

This module exposes lidar detection datasets and datamodules backed by
T4Dataset annotations and sensor metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
import pickle
from typing import Any

from torch.utils.data import DataLoader

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.detection3d import (
    build_detection_dataloader,
    load_detection_data_infos,
    resolve_data_path,
    resolve_sweep_paths,
)
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.annotations import (
    normalize_filter_attributes,
    resolve_detection_class,
)


@dataclass(frozen=True)
class FrameSamplingConfig:
    """Configuration for repeat-factor frame sampling."""

    repeat_sampling_factor: float
    object_bev_range: list[float]
    low_pedestrian_height_threshold: float
    low_pedestrian_bev_range: list[float]
    low_pedestrian_category_name: str = "low_pedestrian"


def coerce_frame_sampling(
    cfg: FrameSamplingConfig | Mapping[str, Any] | None,
) -> FrameSamplingConfig | None:
    """Normalize frame-sampling settings to ``FrameSamplingConfig``."""
    if cfg is None:
        return None
    if isinstance(cfg, FrameSamplingConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return FrameSamplingConfig(**dict(cfg))
    raise TypeError(
        "Expected frame sampling config to be a FrameSamplingConfig, mapping, or None, "
        f"got {type(cfg)!r}."
    )


def compute_frame_sampling_weights(
    data_infos: list[dict[str, Any]],
    class_names: list[str],
    name_mapping: Mapping[str, str],
    frame_sampling: FrameSamplingConfig | None,
    filter_attributes: list[list[str]] | None = None,
    min_num_lidar_points: int = 0,
    use_valid_flag: bool = True,
) -> list[float]:
    """Compute repeat-factor sampling weights for T4 detection samples."""
    if frame_sampling is None:
        return [1.0] * len(data_infos)
    normalized_filter_attributes = normalize_filter_attributes(filter_attributes)

    sampling_categories = [*class_names, frame_sampling.low_pedestrian_category_name]
    category_frame_counts = {category: 0 for category in sampling_categories}
    category_box_counts = {category: 0 for category in sampling_categories}
    frame_categories = []

    for sample in data_infos:
        categories = _sample_sampling_categories(
            sample,
            class_names,
            name_mapping,
            frame_sampling,
            normalized_filter_attributes,
            min_num_lidar_points,
            use_valid_flag,
        )
        frame_categories.append(categories)
        for category, count in categories.items():
            if count <= 0:
                continue
            category_frame_counts[category] += 1
            category_box_counts[category] += count

    total_boxes = sum(category_box_counts.values())
    if total_boxes == 0:
        raise ValueError("Cannot compute frame sampling weights for a dataset with no valid boxes.")

    category_factors = {}
    for category in sampling_categories:
        frame_fraction = category_frame_counts[category] / len(data_infos)
        box_fraction = category_box_counts[category] / total_boxes
        if frame_fraction == 0.0 or box_fraction == 0.0:
            category_factors[category] = 1.0
            continue
        category_fraction = math.sqrt(frame_fraction * box_fraction)
        category_factors[category] = max(
            1.0,
            math.sqrt(frame_sampling.repeat_sampling_factor / category_fraction),
        )

    frame_weights = []
    for categories in frame_categories:
        weight = 1.0
        for category, count in categories.items():
            if count > 0:
                weight = max(weight, category_factors[category])
        frame_weights.append(weight)
    return frame_weights


def _sample_sampling_categories(
    sample: Mapping[str, Any],
    class_names: list[str],
    name_mapping: Mapping[str, str],
    frame_sampling: FrameSamplingConfig,
    filter_attributes: frozenset[tuple[str, str]],
    min_num_lidar_points: int,
    use_valid_flag: bool,
) -> dict[str, int]:
    """Return sampling category counts for one frame."""
    categories = {*class_names, frame_sampling.low_pedestrian_category_name}
    category_counts = {category: 0 for category in categories}

    for instance in sample.get("instances", []):
        mapped_name = resolve_detection_class(
            instance,
            class_names=class_names,
            name_mapping=name_mapping,
            label_to_category=sample.get("label_to_category"),
            filter_attributes=filter_attributes,
            min_num_lidar_points=min_num_lidar_points,
            use_valid_flag=use_valid_flag,
        )
        if mapped_name is None:
            continue
        box = instance.get("bbox_3d")
        if box is None or not _box_center_in_bev_range(box, frame_sampling.object_bev_range):
            continue

        category = mapped_name
        if _is_low_pedestrian(mapped_name, box, frame_sampling):
            category = frame_sampling.low_pedestrian_category_name
        category_counts[category] += 1

    return category_counts


def _is_low_pedestrian(
    mapped_name: str,
    box: list[float],
    frame_sampling: FrameSamplingConfig,
) -> bool:
    """Return whether a mapped box belongs to the low-pedestrian sampling bucket."""
    return (
        mapped_name == "pedestrian"
        and float(box[5]) < frame_sampling.low_pedestrian_height_threshold
        and _box_center_in_bev_range(box, frame_sampling.low_pedestrian_bev_range)
    )


def _box_center_in_bev_range(box: list[float], bev_range: list[float]) -> bool:
    """Check whether a box center is inside ``[x_min, y_min, x_max, y_max]``."""
    x, y = float(box[0]), float(box[1])
    return bev_range[0] <= x <= bev_range[2] and bev_range[1] <= y <= bev_range[3]


class T4Detection3DDataset(Dataset):
    """Load T4Dataset lidar samples for 3D object detection.

    The dataset returns metadata consumed by transform pipelines that load
    lidar points and detection annotations on demand.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        min_num_lidar_points: int = 1,
        use_valid_flag: bool = True,
        frame_sampling: FrameSamplingConfig | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the T4 detection dataset.

        Args:
            data_root: Dataset root directory.
            ann_file: Annotation file path.
            class_names: Ordered detector class names.
            name_mapping: Mapping from dataset labels to detector labels.
            filter_attributes: Raw class-attribute pairs excluded from detection targets.
            min_num_lidar_points: Minimum lidar points required for sampled boxes.
            use_valid_flag: Whether ``bbox_3d_isvalid`` excludes sampled boxes.
            frame_sampling: Optional repeat-factor frame sampling settings.
            dataset_transforms: Optional dataset transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.frame_sampling = frame_sampling

        with open(ann_file, "rb") as file:
            data = pickle.load(file)
        self.data_infos = load_detection_data_infos(data)
        self.frame_weights = compute_frame_sampling_weights(
            self.data_infos,
            self.class_names,
            self.name_mapping,
            self.frame_sampling,
            filter_attributes=filter_attributes,
            min_num_lidar_points=min_num_lidar_points,
            use_valid_flag=use_valid_flag,
        )

    def __len__(self) -> int:
        """Return the number of annotated samples.

        Returns:
            Number of samples available in the annotation file.
        """
        return len(self.data_infos)

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one T4 detection metadata record.

        Args:
            index: Dataset sample index.

        Returns:
            Metadata dictionary consumed by detection transform pipelines.
        """
        sample = self.data_infos[index]
        return {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "name_mapping": self.name_mapping,
            "sample_token": sample["token"],
            "lidar_path": resolve_data_path(self.data_root, sample["lidar_path"]),
            "num_pts_feats": int(sample["lidar_points"].get("num_pts_feats", 5)),
            "sweeps": resolve_sweep_paths(sample, self.data_root),
        }


class T4Detection3DDataModule(DataModule):
    """Create T4Dataset dataloaders for 3D object detection.

    The datamodule configures split-specific datasets and transforms for lidar
    detection experiments on T4Dataset.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        min_num_lidar_points: int = 1,
        use_valid_flag: bool = True,
        train_frame_sampling: FrameSamplingConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the T4 detection datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            class_names: Ordered detector class names.
            name_mapping: Mapping from dataset labels to detector labels.
            filter_attributes: Raw class-attribute pairs excluded from detection targets.
            min_num_lidar_points: Minimum lidar points required for sampled train boxes.
            use_valid_flag: Whether ``bbox_3d_isvalid`` excludes sampled train boxes.
            train_frame_sampling: Optional repeat-factor frame sampling
                settings applied only to the training split.
            **kwargs: Additional base datamodule configuration.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = filter_attributes
        self.min_num_lidar_points = min_num_lidar_points
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
        """Instantiate the dataset for one split.

        Args:
            split: Dataset split name.
            dataset_transforms: Optional transform pipeline for the split.

        Returns:
            Instantiated dataset for the requested split.
        """
        return T4Detection3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            class_names=self.class_names,
            name_mapping=self.name_mapping,
            filter_attributes=self.filter_attributes,
            min_num_lidar_points=self.min_num_lidar_points,
            use_valid_flag=self.use_valid_flag,
            frame_sampling=self.train_frame_sampling if split == "train" else None,
            dataset_transforms=dataset_transforms,
        )

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create a detection dataloader with optional train RFS sampling."""
        return build_detection_dataloader(
            dataset=getattr(self, f"{split}_dataset"),
            dataloader_cfg=getattr(self, f"{split}_dataloader_cfg"),
            is_train=split == "train",
            train_frame_sampling=self.train_frame_sampling,
            collate_fn=self.collate_fn,
        )
