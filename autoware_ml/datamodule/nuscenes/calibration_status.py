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

        The dataset consumes the unified per-frame nuScenes info file and expands
        one calibration record per (frame, camera) pair by iterating each frame's
        ``images`` mapping. Each camera contributes its own ``cam2img``,
        ``lidar2cam`` and distortion parameters against the shared frame lidar.

        Args:
            data_root: Root directory for data files.
            ann_file: Path to the unified annotation file (info.pkl) containing
                per-frame samples under ``data_list`` (or legacy ``infos``).
        """
        super().__init__(**kwargs)
        self.data_root = data_root

        with open(ann_file, "rb") as f:
            data = pickle.load(f)

        if "data_list" in data:
            frames = data["data_list"]
        elif "infos" in data:
            frames = data["infos"]
        else:
            raise ValueError(
                f"Unknown info file format. Expected 'data_list' or 'infos' key, got: {list(data.keys())}"
            )

        # Expand one record per (frame, camera). Each record stores the shared
        # frame lidar path plus the camera's own calibration.
        self.records: list[dict[str, Any]] = []
        for frame in frames:
            if "lidar_points" not in frame:
                raise KeyError("Sample does not contain 'lidar_points' key")
            lidar_path = frame["lidar_points"]["lidar_path"]
            for camera_name, cam_info in frame.get("images", {}).items():
                self.records.append(
                    {
                        "camera_name": camera_name,
                        "cam_info": cam_info,
                        "lidar_path": lidar_path,
                        "frame": frame,
                    }
                )

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of (frame, camera) calibration records.
        """
        return len(self.records)

    def get_data_info(self, idx: int) -> dict[str, Any]:
        """Get sample metadata for a given (frame, camera) record.

        Args:
            idx: Record index.

        Returns:
            Metadata dictionary consumed by transform loaders.
        """
        record = self.records[idx]
        cam_info = record["cam_info"]

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

        calibration_data = CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            lidar_to_camera_transformation=lidar_to_camera_transformation,
            distortion_model=cam_info.get("distortion_model", ""),
        )

        return {
            "img_path": os.path.join(self.data_root, cam_info["img_path"]),
            "lidar_path": os.path.join(self.data_root, record["lidar_path"]),
            "num_pts_feats": 5,
            "calibration_data": calibration_data,
            "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
            "metadata": record["frame"],
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
