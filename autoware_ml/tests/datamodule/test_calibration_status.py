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

"""Unit tests for calibration-status datasets."""

from __future__ import annotations

import pickle

from autoware_ml.datamodule.nuscenes.calibration_status import NuscenesCalibrationStatusDataset
from autoware_ml.datamodule.t4dataset.calibration_status import T4CalibrationStatusDataset
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus


class TestNuscenesCalibrationStatusDataset:
    def test_get_data_info_returns_calibration_data_only(self, tmp_path) -> None:
        ann_file = tmp_path / "infos.pkl"
        sample = {
            "calibration_status_task": True,
            "image": {
                "img_path": "samples/CAM_FRONT/image.jpg",
                "cam2img": [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
                "lidar2cam": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "distortion_coefficients": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
            "lidar_points": {"lidar_path": "samples/LIDAR_TOP/points.bin"},
        }
        with open(ann_file, "wb") as file:
            pickle.dump({"data_list": [sample]}, file)

        dataset = NuscenesCalibrationStatusDataset(data_root=str(tmp_path), ann_file=str(ann_file))

        output = dataset.get_data_info(0)

        assert output["img_path"] == str(tmp_path / "samples/CAM_FRONT/image.jpg")
        assert output["lidar_path"] == str(tmp_path / "samples/LIDAR_TOP/points.bin")
        assert output["num_pts_feats"] == 5
        assert output["gt_calibration_status"] == CalibrationStatus.CALIBRATED.value
        assert output["metadata"] == sample
        assert "camera_matrix" not in output
        assert "distortion_coefficients" not in output
        assert "lidar_to_camera_transformation" not in output
        assert isinstance(output["calibration_data"], CalibrationData)


class TestT4CalibrationStatusDataset:
    def test_get_data_info_returns_calibration_data_only(self, tmp_path) -> None:
        ann_file = tmp_path / "infos.pkl"
        sample = {
            "image": {
                "img_path": "image.jpg",
                "cam2img": [[900.0, 0.0, 300.0], [0.0, 900.0, 200.0], [0.0, 0.0, 1.0]],
                "lidar2cam": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "lidar_points": {"lidar_path": "points.bin"},
        }
        with open(ann_file, "wb") as file:
            pickle.dump({"data_list": [sample]}, file)

        dataset = T4CalibrationStatusDataset(data_root=str(tmp_path), ann_file=str(ann_file))

        output = dataset.get_data_info(0)

        assert output["img_path"] == str(tmp_path / "image.jpg")
        assert output["lidar_path"] == str(tmp_path / "points.bin")
        assert output["num_pts_feats"] == 5
        assert output["gt_calibration_status"] == CalibrationStatus.CALIBRATED.value
        assert output["metadata"] == sample
        assert "camera_matrix" not in output
        assert "distortion_coefficients" not in output
        assert "lidar_to_camera_transformation" not in output
        assert isinstance(output["calibration_data"], CalibrationData)
