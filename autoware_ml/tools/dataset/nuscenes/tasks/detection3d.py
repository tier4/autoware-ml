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

"""Detection3D task annotation generator for NuScenes."""

from typing import Any, Dict

import numpy as np
from pyquaternion import Quaternion

from autoware_ml.tools.dataset.nuscenes.tasks.base import TaskAnnotationGenerator
from autoware_ml.tools.dataset.nuscenes.utils import NUSCENES_NAME_MAPPING


class Detection3DTask(TaskAnnotationGenerator):
    """Task generator for 3D object detection annotations."""

    def process_sample(
        self,
        info_dict: Dict[str, Any],
        nusc: Any,
        sample: Dict[str, Any],
        cam_name: Any = None,
    ) -> Dict[str, Any]:
        """Add detection3d annotations to the info dict.

        Args:
            info_dict: Base info dictionary.
            nusc: NuScenes API instance.
            sample: NuScenes sample dictionary.
            cam_name: Not used for detection3d.

        Returns:
            Updated info dictionary with detection3d annotations.
        """
        if "test" in nusc.version:
            return info_dict

        annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
        if not annotations:
            return info_dict

        lidar_token = sample["data"]["LIDAR_TOP"]
        _, boxes, _ = nusc.get_sample_data(lidar_token)

        l2e_r = info_dict["lidar2ego_rotation"]
        e2g_r = info_dict["ego2global_rotation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
        valid_flag = np.array(
            [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations],
            dtype=bool,
        ).reshape(-1)

        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NUSCENES_NAME_MAPPING:
                names[i] = NUSCENES_NAME_MAPPING[names[i]]
        names = np.array(names)

        gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
        assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"

        info_dict["gt_boxes"] = gt_boxes
        info_dict["gt_names"] = names
        info_dict["gt_velocity"] = velocity.reshape(-1, 2)
        info_dict["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
        info_dict["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
        info_dict["valid_flag"] = valid_flag

        return info_dict
