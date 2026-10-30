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

"""Entity paths of the scene every adapter logs under and every layout shows."""

#: Root of the 3D scene of a sample.
SCENE_PATH = "scene"
#: LiDAR reference layers of the scene.
LIDAR_PATH = f"{SCENE_PATH}/lidar"
#: Predicted layers of the scene.
PREDICTION_PATH = f"{SCENE_PATH}/prediction"
#: Ground-truth layers of the scene.
GROUND_TRUTH_PATH = f"{SCENE_PATH}/ground_truth"
#: Cameras of the scene with their projected layers.
CAMERAS_PATH = f"{SCENE_PATH}/cameras"
#: Sample name and frame extent of the scene.
META_PATH = f"{SCENE_PATH}/meta"
#: Root of a calibration-status sample.
CALIBRATION_STATUS_PATH = "calibration_status"
#: Root of a calibration-status sample previewed without a model.
DATA_CALIBRATION_STATUS_PATH = f"dataset/{CALIBRATION_STATUS_PATH}"
