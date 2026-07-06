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

"""T4Dataset multiview detection dataset and datamodule.

This module contains the T4Dataset multiview adapter used by camera-lidar detectors.
"""

from __future__ import annotations

import os
from typing import Any

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.multiview_detection3d import MultiviewDetection3DDataset
from autoware_ml.transforms.base import TransformsCompose


class T4MultiviewDetection3DDataset(MultiviewDetection3DDataset):
    """Load T4Dataset multiview samples for camera-lidar 3D detection.

    The dataset combines T4 image, lidar, calibration, and detection metadata
    into the common multiview detection interface.
    """


class T4MultiviewDetection3DDataModule(DataModule):
    """Create T4Dataset dataloaders for multiview 3D detection.

    The datamodule configures fusion-oriented dataset adapters and dataloaders
    for T4Dataset multiview experiments.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        class_names: list[str],
        camera_order: list[str],
        name_mapping: dict[str, str] | None = None,
        filter_frames_with_camera_order: bool = True,
        require_image_files: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the T4 multiview detection datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            class_names: Ordered detector class names.
            camera_order: Ordered camera names expected by the model.
            name_mapping: Optional mapping from dataset labels to detector labels.
            filter_frames_with_camera_order: Drop frames missing any camera in
                ``camera_order`` so image loading always sees a complete set.
            require_image_files: Also verify camera image files exist on disk
                while filtering (slower; off by default).
            **kwargs: Additional base datamodule configuration.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.camera_order = camera_order
        self.name_mapping = {} if name_mapping is None else dict(name_mapping)
        self.filter_frames_with_camera_order = filter_frames_with_camera_order
        self.require_image_files = require_image_files

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
        return T4MultiviewDetection3DDataset(
            data_root=self.data_root,
            ann_file=self.ann_files[split],
            class_names=self.class_names,
            camera_order=self.camera_order,
            name_mapping=self.name_mapping,
            filter_frames_with_camera_order=self.filter_frames_with_camera_order,
            require_image_files=self.require_image_files,
            dataset_transforms=dataset_transforms,
        )
