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

"""NuScenes dataset generator.

This module contains reusable logic for generating NuScenes metadata files used
by Autoware-ML dataset preparation commands.
"""

import logging
import os
import pickle
from collections.abc import Sequence
from os import path as osp
from typing import Any

import numpy as np
import numpy.typing as npt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

from autoware_ml.tools.dataset.base import DatasetGenerator
from autoware_ml.tools.dataset.nuscenes.tasks.registry import create_task

logger = logging.getLogger(__name__)

# Canonical nuScenes 10-class detection taxonomy and its label order. This is the
# ordering baked into the on-disk v1.1 info files' ``metainfo["categories"]`` and
# into every ``bbox_label`` / ``bbox_label_3d`` index. Names not in this map are
# stored with label ``-1`` (kept as ignored instances), matching the reference file.
NUSCENES_CATEGORIES: dict[str, int] = {
    "car": 0,
    "truck": 1,
    "trailer": 2,
    "bus": 3,
    "construction_vehicle": 4,
    "bicycle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "traffic_cone": 8,
    "barrier": 9,
}


def _build_transform_matrix(
    translation: Sequence[float], rotation_matrix: npt.NDArray[np.float64]
) -> list[list[float]]:
    """Assemble a 4x4 homogeneous transform from a translation and rotation.

    Args:
        translation: Length-3 translation vector.
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        4x4 transform as nested Python lists.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform.tolist()


def _build_instances(info_dict: dict[str, Any], categories: dict[str, int]) -> list[dict[str, Any]]:
    """Convert detection working fields into v1.1 ``instances`` records.

    Reuses the detection task's ``gt_boxes`` (N,7), ``gt_names``, ``gt_velocity``,
    ``num_lidar_pts``, ``num_radar_pts`` and ``valid_flag``. Names are mapped to
    label indices through ``categories``; names outside the taxonomy get ``-1``
    (mirroring the reference file, which keeps them as ignored boxes).

    Args:
        info_dict: Working info dict after detection-task processing.
        categories: ``{class_name: label_index}`` mapping.

    Returns:
        List of per-object instance dicts, or an empty list when no boxes exist.
    """
    gt_boxes = info_dict.get("gt_boxes")
    if gt_boxes is None:
        return []

    names = info_dict["gt_names"]
    velocity = info_dict["gt_velocity"]
    num_lidar_pts = info_dict["num_lidar_pts"]
    num_radar_pts = info_dict["num_radar_pts"]
    valid_flag = info_dict["valid_flag"]

    instances = []
    for i in range(len(gt_boxes)):
        label = int(categories.get(str(names[i]), -1))
        instances.append(
            {
                "bbox_label": label,
                "bbox_3d": [float(value) for value in gt_boxes[i]],
                "bbox_3d_isvalid": bool(valid_flag[i]),
                "bbox_label_3d": label,
                "num_lidar_pts": int(num_lidar_pts[i]),
                "num_radar_pts": int(num_radar_pts[i]),
                "velocity": [float(velocity[i][0]), float(velocity[i][1])],
            }
        )
    return instances


def _to_unified_record(
    info_dict: dict[str, Any], sample_idx: int, categories: dict[str, int]
) -> dict[str, Any]:
    """Reshape a working info dict into a unified v1.1 per-frame record.

    The unified record is consumed by every nuScenes task (detection3d,
    segmentation3d, and calibration_status), which expand per-camera at load time.

    Args:
        info_dict: Working info dict with pose components, ``images`` and any
            task fields already injected.
        sample_idx: Sequential index of this frame within its split.
        categories: ``{class_name: label_index}`` mapping for instance labels.

    Returns:
        Unified per-frame record with top-level ``data_list`` schema fields.
    """
    l2e_r_mat = Quaternion(info_dict["lidar2ego_rotation"]).rotation_matrix
    e2g_r_mat = Quaternion(info_dict["ego2global_rotation"]).rotation_matrix

    record: dict[str, Any] = {
        "sample_idx": sample_idx,
        "token": info_dict["token"],
        "timestamp": info_dict["timestamp"] / 1e6,
        "ego2global": _build_transform_matrix(info_dict["ego2global_translation"], e2g_r_mat),
        "images": info_dict["images"],
        "lidar_points": {
            "num_pts_feats": info_dict["num_features"],
            "lidar_path": info_dict["lidar_path"],
            "lidar2ego": _build_transform_matrix(info_dict["lidar2ego_translation"], l2e_r_mat),
        },
        "instances": _build_instances(info_dict, categories),
    }
    if "pts_semantic_mask_path" in info_dict:
        record["pts_semantic_mask_path"] = info_dict["pts_semantic_mask_path"]
    if "sweeps" in info_dict:
        record["sweeps"] = info_dict["sweeps"]
    record["scene_token"] = info_dict["scene_token"]
    return record


def _build_lidar_sweeps(
    nusc: NuScenes,
    lidar_sample_data: dict[str, Any],
    current_lidar2ego: npt.NDArray[np.float64],
    current_ego2global: npt.NDArray[np.float64],
    root_path: str,
    max_sweeps: int,
) -> list[dict[str, Any]]:
    """Build historical LiDAR sweep metadata in the current LiDAR frame.

    Args:
        nusc: NuScenes API instance.
        lidar_sample_data: Current frame LIDAR_TOP sample_data record.
        current_lidar2ego: Current LiDAR-to-ego transform.
        current_ego2global: Current ego-to-global transform.
        root_path: Dataset root path used to normalize lidar paths.
        max_sweeps: Maximum number of sweeps including the current frame.

    Returns:
        Historical sweep records ordered from nearest to oldest.
    """
    current_global_from_lidar = current_ego2global @ current_lidar2ego
    current_lidar_from_global = np.linalg.inv(current_global_from_lidar)
    sweeps: list[dict[str, Any]] = []
    sweep_token = lidar_sample_data["prev"]
    for _ in range(max(0, max_sweeps - 1)):
        if not sweep_token:
            break
        sweep_sd = nusc.get("sample_data", sweep_token)
        sweep_cs = nusc.get("calibrated_sensor", sweep_sd["calibrated_sensor_token"])
        sweep_pose = nusc.get("ego_pose", sweep_sd["ego_pose_token"])

        sweep_lidar2ego = np.eye(4, dtype=np.float64)
        sweep_lidar2ego[:3, :3] = Quaternion(sweep_cs["rotation"]).rotation_matrix
        sweep_lidar2ego[:3, 3] = np.asarray(sweep_cs["translation"], dtype=np.float64)
        sweep_ego2global = np.eye(4, dtype=np.float64)
        sweep_ego2global[:3, :3] = Quaternion(sweep_pose["rotation"]).rotation_matrix
        sweep_ego2global[:3, 3] = np.asarray(sweep_pose["translation"], dtype=np.float64)

        current_lidar_from_sweep = current_lidar_from_global @ sweep_ego2global @ sweep_lidar2ego
        sweeps.append(
            {
                "lidar_path": _normalize_path(
                    str(nusc.get_sample_data_path(sweep_token)), root_path
                ),
                "sample_data_token": sweep_token,
                "sensor2lidar_rotation": current_lidar_from_sweep[:3, :3].astype(np.float32),
                "sensor2lidar_translation": current_lidar_from_sweep[:3, 3].astype(np.float32),
                "timestamp": sweep_sd["timestamp"],
            }
        )
        sweep_token = sweep_sd["prev"]
    return sweeps


def get_available_scenes(nusc: NuScenes) -> list[dict[str, Any]]:
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


def _build_camera_info(
    nusc: NuScenes,
    cam_token: str,
    lidar2ego: npt.NDArray[np.float64],
    ego2global_lidar: npt.NDArray[np.float64],
    root_path: str,
) -> dict[str, Any]:
    """Build unified camera info dict for a single camera channel.

    ``lidar2cam`` is composed through BOTH ego poses to account for ego motion
    between the LiDAR and camera capture timestamps (the mmdet3d convention),
    rather than assuming a shared ego frame::

        lidar2cam = inv(cam2ego) @ inv(ego2global_cam) @ ego2global_lidar @ lidar2ego

    where ``ego2global_cam`` is the ego pose at the *camera* timestamp and
    ``ego2global_lidar`` the ego pose at the *LiDAR* timestamp. Ignoring the two
    distinct poses biases the translation by ~0.2 m, which misprojects LiDAR into
    the image (matters for camera-lidar fusion and the calibration GT).

    Args:
        nusc: NuScenes API instance.
        cam_token: Sample data token for the camera.
        lidar2ego: 4x4 LiDAR-to-ego transform (from the LiDAR calibrated sensor).
        ego2global_lidar: 4x4 ego-to-global transform at the LiDAR timestamp.
        root_path: Dataset root path used to normalize the image path.

    Returns:
        Camera info dict with img_path, cam2img [3x3], cam2ego [4x4],
        lidar2cam [4x4], timestamp (seconds), sample_data_token,
        distortion_coefficients, and distortion_model. NuScenes images are
        pre-undistorted so the distortion fields are empty (an empty
        ``distortion_coefficients`` makes ``UndistortImage`` a no-op).
    """
    sd_rec = nusc.get("sample_data", cam_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    _, _, cam_intrinsic = nusc.get_sample_data(cam_token)

    img_path = _normalize_path(str(nusc.get_sample_data_path(sd_rec["token"])), root_path)

    cam2ego = np.eye(4, dtype=np.float64)
    cam2ego[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
    cam2ego[:3, 3] = np.array(cs_record["translation"])

    ego2global_cam = np.eye(4, dtype=np.float64)
    ego2global_cam[:3, :3] = Quaternion(pose_record["rotation"]).rotation_matrix
    ego2global_cam[:3, 3] = np.array(pose_record["translation"])

    # LiDAR frame -> ego(lidar t) -> global -> ego(cam t) -> camera frame.
    lidar2cam = (
        np.linalg.inv(cam2ego) @ np.linalg.inv(ego2global_cam) @ ego2global_lidar @ lidar2ego
    )

    return {
        "img_path": img_path,
        "cam2img": cam_intrinsic.tolist()
        if isinstance(cam_intrinsic, np.ndarray)
        else list(cam_intrinsic),
        "cam2ego": cam2ego.tolist(),
        "lidar2cam": lidar2cam.tolist(),
        "timestamp": sd_rec["timestamp"] / 1e6,
        "sample_data_token": sd_rec["token"],
        "distortion_coefficients": [],
        "distortion_model": "",
    }


class NuScenesDatasetGenerator(DatasetGenerator):
    """Generate NuScenes info files with task-specific annotation payloads.

    The generator reads raw NuScenes samples and injects task-dependent fields
    such as detection or segmentation annotations into exported info files.
    """

    def __init__(self) -> None:
        """Initialize the NuScenes dataset generator.

        The generator keeps the canonical NuScenes camera ordering used when
        image metadata is injected into generated info files.
        """
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
        tasks: Sequence[str],
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
        info_prefix = kwargs.get("info_prefix", "nuscenes")

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
            nusc, train_scenes, val_scenes, test, task_generators, root_path, max_sweeps
        )

        metainfo = {
            "categories": dict(NUSCENES_CATEGORIES),
            "dataset": "nuscenes",
            "version": version,
            "info_version": "1.1",
        }
        os.makedirs(out_dir, exist_ok=True)

        def _write(data_list: list[dict[str, Any]], split: str) -> None:
            data = {"metainfo": metainfo, "data_list": data_list}
            info_path = osp.join(out_dir, f"{info_prefix}_infos_{split}.pkl")
            with open(info_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"wrote {len(data_list)} samples to {info_path}")

        if test:
            logger.info("test sample: {}".format(len(train_nusc_infos)))
            _write(train_nusc_infos, "test")
        else:
            logger.info(
                "train sample: {}, val sample: {}".format(
                    len(train_nusc_infos), len(val_nusc_infos)
                )
            )
            _write(train_nusc_infos, "train")
            _write(val_nusc_infos, "val")

    def _fill_trainval_infos(
        self,
        nusc: NuScenes,
        train_scenes: set[str],
        val_scenes: set[str],
        test: bool,
        task_generators: Sequence[Any],
        root_path: str,
        max_sweeps: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate the train/val infos from the raw data.

        Args:
            nusc: Dataset class in the nuScenes dataset.
            train_scenes: Set of training scene tokens.
            val_scenes: Set of validation scene tokens.
            test: Whether use the test mode.
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
                "scene_token": sample["scene_token"],
                "images": {},
                "lidar2ego_translation": cs_record["translation"],
                "lidar2ego_rotation": cs_record["rotation"],
                "ego2global_translation": pose_record["translation"],
                "ego2global_rotation": pose_record["rotation"],
                "timestamp": sample["timestamp"],
            }

            lidar2ego = np.eye(4, dtype=np.float64)
            lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
            ego2global_lidar = np.eye(4, dtype=np.float64)
            ego2global_lidar[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
            ego2global_lidar[:3, 3] = np.array(info["ego2global_translation"])
            info["sweeps"] = _build_lidar_sweeps(
                nusc, sd_rec, lidar2ego, ego2global_lidar, root_path, max_sweeps
            )

            for cam in self.camera_types:
                cam_token = sample["data"][cam]
                info["images"][cam] = _build_camera_info(
                    nusc, cam_token, lidar2ego, ego2global_lidar, root_path
                )

            for task_gen in task_generators:
                info = task_gen.process_sample(info, nusc, sample)

            if sample["scene_token"] in train_scenes:
                record = _to_unified_record(info, len(train_nusc_infos), NUSCENES_CATEGORIES)
                train_nusc_infos.append(record)
            else:
                record = _to_unified_record(info, len(val_nusc_infos), NUSCENES_CATEGORIES)
                val_nusc_infos.append(record)

        return train_nusc_infos, val_nusc_infos
