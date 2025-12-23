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

"""Calibration status task annotation generator for NuScenes."""

import os
from typing import Any, Dict

import numpy as np

from autoware_ml.tools.dataset.nuscenes.tasks.base import TaskAnnotationGenerator


class CalibrationStatusTask(TaskAnnotationGenerator):
    """Task generator for calibration status annotations.

    Creates separate sample entries for each camera, extracting calibration data
    including camera intrinsics, distortion coefficients, and lidar-to-camera transforms.
    """

    def process_sample(
        self,
        info_dict: Dict[str, Any],
        nusc: Any,
        sample: Dict[str, Any],
        cam_name: Any = None,
    ) -> Dict[str, Any]:
        """Add calibration_status annotations to the info dict.

        For calibration_status, we create separate samples for each camera.
        This method adds a special marker that the generator will use to expand
        into multiple samples.

        Args:
            info_dict: Base info dictionary.
            nusc: NuScenes API instance.
            sample: NuScenes sample dictionary.
            cam_name: Not used directly, but we process all cameras.

        Returns:
            Updated info dictionary with calibration_status annotations.
        """
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

        calibration_samples = []
        for cam in camera_types:
            if cam not in info_dict["cams"]:
                continue

            cam_info = info_dict["cams"][cam]
            cam_token = sample["data"][cam]
            cam_path, _, _ = nusc.get_sample_data(cam_token)
            cam_path = str(cam_path)

            root_path = nusc.dataroot
            if os.path.isabs(cam_path):
                cam_path_abs = os.path.abspath(cam_path)
                root_path_abs = os.path.abspath(root_path)
                if cam_path_abs.startswith(root_path_abs):
                    cam_path = os.path.relpath(cam_path_abs, root_path_abs)

            root_path_normalized = os.path.basename(root_path)
            if cam_path.startswith(f"data/{root_path_normalized}/"):
                cam_path = cam_path[len(f"data/{root_path_normalized}/") :]
            elif cam_path.startswith(f"{root_path_normalized}/"):
                cam_path = cam_path[len(f"{root_path_normalized}/") :]

            sd_rec = nusc.get("sample_data", cam_token)
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])

            cam_intrinsic = cam_info.get("cam_intrinsic")
            if cam_intrinsic is None:
                continue

            sensor2lidar_rotation = cam_info.get("sensor2lidar_rotation")
            sensor2lidar_translation = cam_info.get("sensor2lidar_translation")

            if sensor2lidar_rotation is None or sensor2lidar_translation is None:
                continue

            R = np.array(sensor2lidar_rotation)
            T = np.array(sensor2lidar_translation)

            lidar2cam_rotation = R.T
            lidar2cam_translation = -R.T @ T.reshape(3, 1) if T.ndim == 1 else -R.T @ T
            if lidar2cam_translation.ndim == 2:
                lidar2cam_translation = lidar2cam_translation.flatten()

            lidar2cam = np.eye(4, dtype=np.float32)
            lidar2cam[:3, :3] = lidar2cam_rotation
            lidar2cam[:3, 3] = lidar2cam_translation

            distortion_coefficients = cs_record.get("camera_distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
            if len(distortion_coefficients) != 5:
                distortion_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0]

            calibration_sample = {
                "image": {
                    "img_path": str(cam_path),
                    "cam2img": (
                        cam_intrinsic.tolist()
                        if isinstance(cam_intrinsic, np.ndarray)
                        else cam_intrinsic
                    ),
                    "lidar2cam": lidar2cam.tolist(),
                    "distortion_coefficients": distortion_coefficients,
                    "sample_data_token": cam_token,
                },
                "lidar_points": {
                    "lidar_path": info_dict["lidar_path"],
                    "sample_data_token": sample["data"]["LIDAR_TOP"],
                },
                "calibration_status_task": True,
                "camera_name": cam,
            }
            calibration_samples.append(calibration_sample)

        info_dict["calibration_status_samples"] = calibration_samples
        return info_dict
