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
    """Structured representation of camera calibration data.
    This class holds all the necessary calibration information for camera-LiDAR
    coordinate transformations and image processing transforms.
    """

    camera_matrix: npt.NDArray[np.float32]  # Original camera intrinsic matrix (3x3)
    distortion_coefficients: npt.NDArray[np.float32]  # Camera distortion coefficients
    lidar_to_camera_transformation: npt.NDArray[np.float32]
    # Updated camera matrix after image processing (undistortion, cropping, scaling)
    # This matrix should be used for 3D->2D projection after any image transformations
    # to ensure geometric consistency between the processed image and 3D point projections
    new_camera_matrix: Optional[npt.NDArray[np.float32]] = None

    def __post_init__(self):
        """Initialize new_camera_matrix if not provided.
        Sets new_camera_matrix to a copy of camera_matrix if it's None.
        """
        if self.new_camera_matrix is None:
            self.new_camera_matrix = self.camera_matrix.copy()


class T4CalibrationStatusDataset(Dataset):
    """Dataset for T4 Calibration Status using the info.pkl structure.

    This dataset loads calibration status data from T4 dataset format.
    Each sample contains comprehensive sensor and calibration information including
    camera images, lidar point clouds, and geometric transformations.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the T4 Calibration Status Dataset.

        Args:
            data_root: Root directory for data files.
            ann_file: Path to the annotation file (info.pkl) containing sample data.
        """
        super().__init__(**kwargs)
        self.data_root = data_root

        with open(ann_file, "rb") as f:
            self.data_infos = pickle.load(f)

    def __len__(self) -> int:
        """Get dataset length.
        Returns:
            Dataset length.
        """
        return len(self.data_infos["data_list"])

    def _get_input_dict(self, idx: int) -> Dict[str, Any]:
        """Get input dictionary for a given index.

        Args:
            idx: Sample index.

        Returns:
            Input dictionary.
        """
        input_dict = dict()

        image_path: str = self.data_infos["data_list"][idx]["image"]["img_path"]
        lidar_path: str = self.data_infos["data_list"][idx]["lidar_points"]["lidar_path"]

        num_pts_feats = 5

        image = cv2.imread(os.path.join(self.data_root, image_path))
        points = np.fromfile(os.path.join(self.data_root, lidar_path), dtype=np.float32).reshape(
            -1, num_pts_feats
        )
        calibration_data = self._load_calibration_data(self.data_infos["data_list"][idx])

        input_dict["img"] = image
        input_dict["points"] = points
        input_dict["calibration_data"] = calibration_data
        input_dict["gt_calibration_status"] = CalibrationStatus.CALIBRATED.value
        input_dict["metadata"] = self.data_infos["data_list"][idx]

        return input_dict

    def _load_calibration_data(self, sample: dict) -> CalibrationData:
        if "image" not in sample:
            raise KeyError("Sample does not contain 'image' key")

        if "lidar_points" not in sample:
            raise KeyError("Sample does not contain 'lidar_points' key")

        cam_info = sample["image"]

        # Validate camera
        camera_matrix = cam_info.get("cam2img", None)
        if camera_matrix is None:
            raise ValueError("Camera matrix (cam2img) is missing")
        camera_matrix = np.array(camera_matrix)

        if camera_matrix.shape != (3, 3):
            raise ValueError(f"Camera matrix must be 3x3, got shape {camera_matrix.shape}")

        # Validate Lidar
        lidar_to_camera_transformation = cam_info.get("lidar2cam", None)
        if lidar_to_camera_transformation is None:
            raise ValueError("lidar_to_camera_transformation is missing")

        lidar_to_camera_transformation = np.array(lidar_to_camera_transformation)
        if lidar_to_camera_transformation.shape != (4, 4):
            raise ValueError(
                f"lidar_to_camera_transformation must be 4x4, got shape {lidar_to_camera_transformation.shape}"
            )

        # Validate distortion coefficients
        distortion_coefficients = cam_info.get("distortion_coefficients", None)
        if distortion_coefficients is None:
            raise ValueError("distortion_coefficients is missing")
        distortion_coefficients = np.array(distortion_coefficients)
        if distortion_coefficients.shape != (5,):
            raise ValueError(
                f"distortion_coefficients must be 5, got shape {distortion_coefficients.shape}"
            )

        return CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            lidar_to_camera_transformation=lidar_to_camera_transformation,
        )


class T4CalibrationDataModule(DataModule):
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

    def _create_dataset(
        self, split: str, dataset_transforms: Optional[TransformsCompose] = None
    ) -> Dataset:
        """Create dataset for a specific split.

        Args:
            split: Dataset split name ("train", "val", "test", "predict").
            dataset_transforms: TransformsCompose for the dataset.

        Returns:
            Dataset instance for the split.
        """
        ann_file = self.ann_files.get(split)

        # Create dataset with transforms
        dataset = T4CalibrationStatusDataset(
            data_root=self.data_root,
            ann_file=ann_file,
            dataset_transforms=dataset_transforms,
        )

        return dataset
