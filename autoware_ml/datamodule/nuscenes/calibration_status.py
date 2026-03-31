# Copyright 2025 TIER IV, Inc.
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

"""NuScenes calibration-status dataset and datamodule."""

import os
import pickle
from typing import Any

import numpy as np

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus


class NuscenesCalibrationStatusDataset(Dataset):
    """Dataset for NuScenes calibration-status metadata samples."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the NuScenes Calibration Status Dataset.

        Args:
            data_root: Root directory for data files.
            ann_file: Path to the annotation file (info.pkl) containing sample data.
        """
        super().__init__(**kwargs)
        self.data_root = data_root

        with open(ann_file, "rb") as f:
            data = pickle.load(f)

        if "infos" in data:
            self.data_infos = data["infos"]
        elif "data_list" in data:
            self.data_infos = data["data_list"]
        else:
            raise ValueError(
                f"Unknown info file format. Expected 'infos' or 'data_list' key, got: {list(data.keys())}"
            )

        self.data_infos = [
            info for info in self.data_infos if info.get("calibration_status_task", False)
        ]

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Dataset length.
        """
        return len(self.data_infos)

    def get_data_info(self, idx: int) -> dict[str, Any]:
        """Get sample metadata for a given index.

        Args:
            idx: Sample index.

        Returns:
            Metadata dictionary consumed by transform loaders.
        """
        sample = self.data_infos[idx]

        if "image" not in sample:
            raise KeyError("Sample does not contain 'image' key")
        if "lidar_points" not in sample:
            raise KeyError("Sample does not contain 'lidar_points' key")

        image_path: str = sample["image"]["img_path"]
        lidar_path: str = sample["lidar_points"]["lidar_path"]

        cam_info = sample["image"]

        camera_matrix = cam_info.get("cam2img", None)
        if camera_matrix is None:
            raise ValueError("Camera matrix (cam2img) is missing")
        camera_matrix = np.array(camera_matrix, dtype=np.float32)

        if camera_matrix.shape != (3, 3):
            raise ValueError(f"Camera matrix must be 3x3, got shape {camera_matrix.shape}")

        lidar_to_camera_transformation = cam_info.get("lidar2cam", None)
        if lidar_to_camera_transformation is None:
            raise ValueError("lidar_to_camera_transformation is missing")

        lidar_to_camera_transformation = np.array(lidar_to_camera_transformation, dtype=np.float32)
        if lidar_to_camera_transformation.shape != (4, 4):
            raise ValueError(
                f"lidar_to_camera_transformation must be 4x4, got shape {lidar_to_camera_transformation.shape}"
            )

        distortion_coefficients = cam_info.get("distortion_coefficients", None)
        if distortion_coefficients is None:
            raise ValueError("distortion_coefficients is missing")
        distortion_coefficients = np.array(distortion_coefficients, dtype=np.float32)
        if distortion_coefficients.shape != (5,):
            raise ValueError(
                f"distortion_coefficients must be 5, got shape {distortion_coefficients.shape}"
            )

        calibration_data = CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            lidar_to_camera_transformation=lidar_to_camera_transformation,
        )

        return {
            "img_path": os.path.join(self.data_root, image_path),
            "lidar_path": os.path.join(self.data_root, lidar_path),
            "num_pts_feats": 5,
            "calibration_data": calibration_data,
            "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
            "metadata": sample,
        }


class NuscenesCalibrationDataModule(DataModule):
    """DataModule for NuScenes Calibration Status dataset.

    This DataModule provides train/val/test/predict dataloaders for the
    NuScenes calibration status classification task.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        **kwargs: Any,
    ):
        """Initialize NuScenes Calibration DataModule.

        Args:
            data_root: Root directory for data files.
            train_ann_file: Path to training annotation file.
            val_ann_file: Path to validation annotation file.
            test_ann_file: Path to test annotation file.
        """
        super().__init__(**kwargs)
        self.data_root = data_root

        def resolve_ann_file(ann_file: str) -> str:
            if os.path.isabs(ann_file):
                return ann_file
            return os.path.join(data_root, ann_file)

        self.ann_files = {
            "train": resolve_ann_file(train_ann_file),
            "val": resolve_ann_file(val_ann_file),
            "test": resolve_ann_file(test_ann_file),
            "predict": resolve_ann_file(test_ann_file),
        }

    def _create_dataset(
        self, split: str, dataset_transforms: TransformsCompose | None = None
    ) -> Dataset:
        """Create dataset for a specific split.

        Args:
            split: Dataset split name ("train", "val", "test", "predict").
            dataset_transforms: TransformsCompose for the dataset.

        Returns:
            Dataset instance for the split.
        """
        ann_file = self.ann_files.get(split)

        dataset = NuscenesCalibrationStatusDataset(
            data_root=self.data_root,
            ann_file=ann_file,
            dataset_transforms=dataset_transforms,
        )

        return dataset
