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

import os
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import cv2
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

    lidar1_to_lidar2_transform: npt.NDArray[np.float32]


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
        lidar_source1: str,
        lidar_source2: str,
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
        self.lidar_source1 = lidar_source1
        self.lidar_source2 = lidar_source2

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

        lidar_path: str = self.data_infos["data_list"][index]["lidar_points"]["lidar_path"]
        lidar_info_path = lidar_path # TODO: change the path from ".../LIDAR_CONCAT/{ID}.pcd.bin" to ".../LIDAR_CONCAT_INFO/{ID}.json" 

        num_pts_feats = 5

        lidar_points = np.fromfile(os.path.join(self.data_root, lidar_path), dtype=np.float32).reshape(
            -1, num_pts_feats
        )
        lidar1_points, lidar2_points = _split_points(self, lidar_points, lidar_info_path)
        calibration_data = self._load_calibration_data(self.data_infos["data_list"][index])

        input_dict["lidar1_points"] = lidar1_points
        input_dict["lidar2_points"] = lidar2_points
        input_dict["calibration_data"] = calibration_data
        input_dict["gt_calibration_status"] = CalibrationStatus.CALIBRATED.value
        input_dict["metadata"] = self.data_infos["data_list"][index]

        return input_dict

    def _split_points(self, lidar_points, info_path : str):
        # TODO read the json file located in lidar_info_path to retrieve the points index for lidar1 and lidar2
        # TODO split the lidar_points into lidar1_points and lidar2_points based on the indexes from the json file
        None

    def _load_calibration_data(self, sample: dict) -> CalibrationData:
        if "lidar_points" not in sample:
            raise KeyError("Sample does not contain 'lidar_points' key")
        if "lidar_sources" not in sample:
            raise KeyError("Sample does not contain 'lidar_sources' key")

        origin_to_lidar1 = sample["lidar_sources"][self.lidar_source1]
        origin_to_lidar2 = sample["lidar_sources"][self.lidar_source2]

        # Validate Lidar
        points_origin = sample["lidar_points"].get("lidar_pose", None)
        if points_origin is None:
            raise ValueError("concatenated pointcloud origin is missing")
        # TODO validate the transforms are in the correct format

        # TODO calculate the transform lidar1 to lidar2
        transform = []

        return CalibrationData(
          lidar1_to_lidar2_transform=transform
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
        lidar_source1: str,
        lidar_source2: str,
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
        self.lidar_source1 = lidar_source1
        self.lidar_source2 = lidar_source2

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
            lidar_source1=self.lidar_source1,
            lidar_source2=self.lidar_source2
        )

        return dataset
