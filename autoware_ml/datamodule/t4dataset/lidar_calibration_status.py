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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIN>D, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.transforms import TransformsCompose


class CalibrationStatus(Enum):
    """Enumeration for calibration status."""

    MISCALIBRATED = 0
    CALIBRATED = 1


@dataclass
class CalibrationData:
    """Structured representation of lidar calibration data.
    This class holds all the necessary calibration information for LiDAR-LiDAR
    coordinate transformations
    """

    ground_truth_baselink_to_lidar: Dict[str, npt.NDArray[np.float32]]
    noise_baselink_to_lidar: Dict[str, npt.NDArray[np.float32]]


class T4LidarCalibrationStatusDataset(Dataset):
    """Dataset for T4 Calibration Status using the info.pkl structure.

    This dataset loads calibration status data from T4 dataset format.
    Each sample contains comprehensive sensor and calibration information including
    camera images, lidar point clouds, and geometric transformations.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        lidar_sources: list[str],
        **kwargs: Any,
    ) -> None:
        """Initialize the T4 Calibration Status Dataset.

        Args:
            data_root: Root directory for data files.
            ann_file: Path to the annotation file (info.pkl) containing sample data.
            lidar_sources:
        """
        super().__init__(**kwargs)
        self.data_root = data_root

        with open(ann_file, "rb") as f:
            self.data_infos = pickle.load(f)

        self.data_infos["data_list"] = self.data_infos["data_list"]
        self.lidar_sources = lidar_sources

    def __len__(self) -> int:
        """Get dataset length.
        Returns:
            Dataset length.
        """
        return len(self.data_infos["data_list"])

    def _get_input_dict(self, index: int) -> Dict[str, Any]:
        """Get input dictionary for a given index.

        Args:
            idx: Sample index.

        Returns:
            Input dictionary.
        """
        input_dict = dict()

        sample = self.data_infos["data_list"][index]
        lidar_path: str = sample["lidar_points"]["lidar_path"]
        lidar_info_path = lidar_path.replace("LIDAR_CONCAT", "LIDAR_CONCAT_INFO").replace(
            ".pcd.bin", ".json"
        )

        num_pts_feats = 5

        lidar_points = np.fromfile(
            os.path.join(self.data_root, lidar_path), dtype=np.float32
        ).reshape(-1, num_pts_feats)

        points_per_lidar = {
            lidar_source: self._extract_lidar_points(
                lidar_points, lidar_info_path, sample["lidar_sources"][lidar_source]["sensor_token"]
            )
            for lidar_source in self.lidar_sources
        }
        calibration_data = self._load_calibration_data(sample)

        input_dict["points_per_lidar"] = points_per_lidar
        input_dict["calibration_data"] = calibration_data
        input_dict["gt_calibration_status"] = CalibrationStatus.CALIBRATED.value

        return input_dict

    def _extract_lidar_points(
        self, lidar_points: npt.NDArray[np.float32], info_path: str, token: str
    ) -> npt.NDArray[np.float32]:
        with open(os.path.join(self.data_root, info_path), "r") as f:
            info = json.load(f)

        points = None
        for source in info["sources"]:
            if source["sensor_token"] == token:
                idx_begin = source["idx_begin"]
                length = source["length"]
                points = lidar_points[idx_begin : idx_begin + length]

        if points is None:
            raise ValueError(f"Could not find sensor {token} in {info_path}.")

        return points

    def _load_calibration_data(self, sample: dict) -> CalibrationData:
        if "lidar_points" not in sample:
            raise KeyError("Sample does not contain 'lidar_points' key")
        if "lidar_sources" not in sample:
            raise KeyError("Sample does not contain 'lidar_sources' key")

        def get_transform(source_dict: dict):
            rotation = np.array(source_dict["rotation"])
            translation = np.array(source_dict["translation"])
            if rotation.shape != (3, 3):
                raise ValueError(f"Rotation matrix must be 3x3, got {rotation.shape}")
            if translation.shape != (3,):
                raise ValueError(f"Translation vector must be (3,), got {translation.shape}")

            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            return transform

        ground_truth_baselink_to_lidar = {
            lidar_source: get_transform(sample["lidar_sources"][lidar_source])
            for lidar_source in self.lidar_sources
        }
        empty_noise = {
            lidar_source: np.eye(4, dtype=np.float32) for lidar_source in self.lidar_sources
        }
        return CalibrationData(
            ground_truth_baselink_to_lidar=ground_truth_baselink_to_lidar,
            noise_baselink_to_lidar=empty_noise,
        )


class T4LidarCalibrationDataModule(DataModule):
    """DataModule for T4 Calibration Status dataset.

    This DataModule provides train/val/test/predict dataloaders for the
    T4 calibration status classification task.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        lidar_sources: list[str],
        **kwargs: Any,
    ):
        """Initialize T4 Calibration DataModule.

        Args:
            data_root: Root directory for data files.
            train_ann_file: Path to training annotation file.
            val_ann_file: Path to validation annotation file.
            test_ann_file: Path to test annotation file.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.ann_files = {
            "train": os.path.join(data_root, train_ann_file),
            "val": os.path.join(data_root, val_ann_file),
            "test": os.path.join(data_root, test_ann_file),
            "predict": os.path.join(data_root, test_ann_file),
        }
        self.lidar_sources = lidar_sources

    def _create_dataset(
        self, split: str, transforms: Optional[TransformsCompose] = None
    ) -> Dataset:
        """Create dataset for a specific split.

        Args:
            split: Dataset split name ("train", "val", "test", "predict").
            dataset_transforms: TransformsCompose for the dataset.

        Returns:
            Dataset instance for the split.
        """
        ann_file = self.ann_files.get(split)
        if ann_file is None:
            raise KeyError(f"could not get ann_file for key '{split}'")

        # Create dataset with transforms
        dataset = T4LidarCalibrationStatusDataset(
            data_root=self.data_root,
            ann_file=ann_file,
            lidar_sources=self.lidar_sources,
            dataset_transforms=transforms,
        )

        return dataset
