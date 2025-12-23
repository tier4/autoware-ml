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

"""NuScenes dataset generator."""

import logging
import os
import pickle
from os import path as osp
from typing import Any, Dict, List, Set

import numpy as np
from pyquaternion import Quaternion

from autoware_ml.tools.dataset.base import DatasetGenerator
from autoware_ml.tools.dataset.nuscenes.tasks import create_task
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

logger = logging.getLogger(__name__)


def get_available_scenes(nusc: NuScenes) -> List[Dict[str, Any]]:
    """Get available scenes from the input nuscenes class.

    Args:
        nusc: Dataset class in the nuScenes dataset.

    Returns:
        List of basic information for the available scenes.
    """
    available_scenes = []
    logger.info("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
            if not os.path.isfile(lidar_path) and not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    logger.info("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def _normalize_path(path: str, root_path: str) -> str:
    """Normalize a path to be relative to root_path.

    Args:
        path: Absolute or relative path to normalize.
        root_path: Root path to make the path relative to.

    Returns:
        Path relative to root_path.
    """
    path = str(path)
    root_path = os.path.abspath(root_path)

    if os.path.isabs(path):
        path_abs = os.path.abspath(path)
        if path_abs.startswith(root_path):
            rel_path = os.path.relpath(path_abs, root_path)
            return rel_path

    if os.getcwd() in path:
        path = path.split(f"{os.getcwd()}/")[-1]

    root_path_normalized = os.path.basename(root_path)
    if path.startswith(f"data/{root_path_normalized}/"):
        path = path[len(f"data/{root_path_normalized}/") :]
    elif path.startswith(f"{root_path_normalized}/"):
        path = path[len(f"{root_path_normalized}/") :]

    return path


def obtain_sensor2top(
    nusc: NuScenes,
    sensor_token: str,
    l2e_t: np.ndarray,
    l2e_r_mat: np.ndarray,
    e2g_t: np.ndarray,
    e2g_r_mat: np.ndarray,
    sensor_type: str = "lidar",
) -> Dict[str, Any]:
    """Obtain the info with RT matrix from general sensor to Top LiDAR.

    Args:
        nusc: Dataset class in the nuScenes dataset.
        sensor_token: Sample data token corresponding to the specific sensor type.
        l2e_t: Translation from lidar to ego.
        l2e_r_mat: Rotation matrix from lidar to ego.
        e2g_t: Translation from ego to global.
        e2g_r_mat: Rotation matrix from ego to global.
        sensor_type: Sensor to calibrate.

    Returns:
        Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    data_path = _normalize_path(data_path, nusc.dataroot)
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T
    sweep["sensor2lidar_translation"] = T
    return sweep


class NuScenesDatasetGenerator(DatasetGenerator):
    """NuScenes dataset generator with task-based annotation injection."""

    def __init__(self) -> None:
        """Initialize NuScenes dataset generator."""
        self.camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

    def generate(
        self,
        root_path: str,
        out_dir: str,
        tasks: List[str],
        **kwargs: Any,
    ) -> None:
        """Generate NuScenes dataset info files.

        Args:
            root_path: Root path of the NuScenes dataset.
            out_dir: Output directory for info files.
            tasks: List of task names to generate annotations for.
            **kwargs: Dataset-specific arguments:
                - version: NuScenes version (default: 'v1.0-trainval')
                - max_sweeps: Max number of sweeps (default: 10)
                - info_prefix: Prefix for info files (default: 'nuscenes')
        """
        # assert if kwargs contains version, max_sweeps, info_prefix
        version = kwargs.get("version", "v1.0-trainval")
        max_sweeps = kwargs.get("max_sweeps", 10)
        tasks_joined = "_".join(sorted(tasks))
        info_prefix = kwargs.get("info_prefix", f"nuscenes_{tasks_joined}")

        logger.info(f"version: {version}, max_sweeps: {max_sweeps}, info_prefix: {info_prefix}")
        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

        available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        if version not in available_vers:
            raise ValueError(f"Unsupported version: {version}. Available: {available_vers}")

        if version == "v1.0-trainval":
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == "v1.0-test":
            train_scenes = splits.test
            val_scenes = []
        elif version == "v1.0-mini":
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val

        available_scenes = get_available_scenes(nusc)
        available_scene_names = [s["name"] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set(
            [available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes]
        )
        val_scenes = set(
            [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
        )

        test = "test" in version
        if test:
            logger.info("test scene: {}".format(len(train_scenes)))
        else:
            logger.info("train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes)))

        task_generators = [create_task(task_name) for task_name in tasks]

        train_nusc_infos, val_nusc_infos = self._fill_trainval_infos(
            nusc, train_scenes, val_scenes, test, max_sweeps, task_generators, root_path
        )

        metadata = dict(version=version)
        os.makedirs(out_dir, exist_ok=True)

        if test:
            logger.info("test sample: {}".format(len(train_nusc_infos)))
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = osp.join(out_dir, f"{info_prefix}_infos_test.pkl")
            with open(info_path, "wb") as f:
                pickle.dump(data, f)
        else:
            logger.info(
                "train sample: {}, val sample: {}".format(
                    len(train_nusc_infos), len(val_nusc_infos)
                )
            )
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
            with open(info_path, "wb") as f:
                pickle.dump(data, f)
            data["infos"] = val_nusc_infos
            info_val_path = osp.join(out_dir, f"{info_prefix}_infos_val.pkl")
            with open(info_val_path, "wb") as f:
                pickle.dump(data, f)

    def _fill_trainval_infos(
        self,
        nusc: NuScenes,
        train_scenes: Set[str],
        val_scenes: Set[str],
        test: bool,
        max_sweeps: int,
        task_generators: List[Any],
        root_path: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate the train/val infos from the raw data.

        Args:
            nusc: Dataset class in the nuScenes dataset.
            train_scenes: Set of training scene tokens.
            val_scenes: Set of validation scene tokens.
            test: Whether use the test mode.
            max_sweeps: Max number of sweeps.
            task_generators: List of task annotation generators.

        Returns:
            Tuple of (train_infos, val_infos).
        """
        train_nusc_infos = []
        val_nusc_infos = []

        total_samples = len(nusc.sample)
        for idx, sample in enumerate(nusc.sample):
            if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
                logger.info(f"Processing sample {idx + 1}/{total_samples}")
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

            if not os.path.exists(lidar_path):
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")

            lidar_path = _normalize_path(lidar_path, root_path)

            info = {
                "lidar_path": lidar_path,
                "num_features": 5,
                "token": sample["token"],
                "sweeps": [],
                "cams": dict(),
                "lidar2ego_translation": cs_record["translation"],
                "lidar2ego_rotation": cs_record["rotation"],
                "ego2global_translation": pose_record["translation"],
                "ego2global_rotation": pose_record["rotation"],
                "timestamp": sample["timestamp"],
            }

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            for cam in self.camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info["cams"].update({cam: cam_info})

            sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            sweeps = []
            while len(sweeps) < max_sweeps:
                if not sd_rec["prev"] == "":
                    sweep = obtain_sensor2top(
                        nusc,
                        sd_rec["prev"],
                        l2e_t,
                        l2e_r_mat,
                        e2g_t,
                        e2g_r_mat,
                        "lidar",
                    )
                    sweeps.append(sweep)
                    sd_rec = nusc.get("sample_data", sd_rec["prev"])
                else:
                    break
            info["sweeps"] = sweeps

            for task_gen in task_generators:
                info = task_gen.process_sample(info, nusc, sample)

            if "calibration_status_samples" in info:
                calibration_samples = info.pop("calibration_status_samples")
                for calib_sample in calibration_samples:
                    calib_info = info.copy()
                    calib_info.update(calib_sample)
                    if sample["scene_token"] in train_scenes:
                        train_nusc_infos.append(calib_info)
                    else:
                        val_nusc_infos.append(calib_info)
            else:
                if sample["scene_token"] in train_scenes:
                    train_nusc_infos.append(info)
                else:
                    val_nusc_infos.append(info)

        return train_nusc_infos, val_nusc_infos
