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

from autoware_ml.datamodule.base import Dataset
from autoware_ml.datamodule.common.multiview_detection3d import (
    MultiviewDetection3DDataModule,
    MultiviewDetection3DDataset,
)
from autoware_ml.transforms.base import TransformsCompose


class T4MultiviewDetection3DDataset(MultiviewDetection3DDataset):
    """Load T4Dataset multiview samples for camera-lidar 3D detection.

    The dataset combines T4 image, lidar, calibration, and detection metadata
    into the common multiview detection interface.
    """


class T4MultiviewDetection3DDataModule(MultiviewDetection3DDataModule):
    """Create T4Dataset dataloaders for multiview 3D detection.

    The datamodule configures fusion-oriented dataset adapters and dataloaders
    for T4Dataset multiview experiments.
    """

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
