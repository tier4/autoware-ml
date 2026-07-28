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

"""Shared multiview detection3d dataset utilities.

This module contains reusable dataset and collation helpers for camera-lidar
3D detection tasks that consume synchronized multiview inputs.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.detection3d import (
    build_label_to_category,
    load_detection_data_infos,
)
from autoware_ml.datamodule.samplers import GroupStreamingSampler

logger = logging.getLogger(__name__)


def build_streaming_dataloader(
    dataset: Dataset,
    dataloader_cfg: Any,
    collate_fn: Any,
    *,
    shuffle_scenes: bool,
) -> DataLoader:
    """Build a dataloader driven by scene-streaming sampling.

    Every dataloader lane walks whole scenes frame by frame, so stateful
    temporal models see contiguous streams in training and evaluation alike.
    The :class:`GroupStreamingSampler` owns the sample order, so the
    dataloader configuration must not request shuffling.

    Args:
        dataset: Dataset exposing ``scene_index_groups()``.
        dataloader_cfg: Split dataloader configuration.
        collate_fn: Batch collation callable.
        shuffle_scenes: Whether the scene order is reshuffled every epoch.

    Returns:
        Streaming dataloader for the split.
    """
    dataloader_kwargs = dataloader_cfg.to_dataloader_kwargs()
    if dataloader_kwargs.pop("shuffle"):
        raise ValueError(
            "Streaming datamodules own the sample order; set dataloader shuffle to false."
        )
    sampler = GroupStreamingSampler(
        dataset, batch_size=dataloader_cfg.batch_size, shuffle=shuffle_scenes
    )
    return DataLoader(dataset=dataset, sampler=sampler, collate_fn=collate_fn, **dataloader_kwargs)


class MultiviewDetection3DDataModule(DataModule):
    """Shared datamodule scaffolding for multiview 3D detection datasets.

    Subclasses provide ``_create_dataset``; annotation-file resolution and the
    scene-streaming dataloader policy live here so every multiview dataset
    behaves identically.
    """

    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: str,
        class_names: list[str],
        camera_order: list[str],
        name_mapping: dict[str, str] | None = None,
        filter_frames_with_camera_order: bool = True,
        require_image_files: bool = False,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the multiview detection datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path.
            val_ann_file: Validation annotation file path.
            test_ann_file: Test annotation file path.
            class_names: Ordered detector class names.
            camera_order: Ordered camera names expected by the model.
            name_mapping: Optional mapping from dataset labels to detector labels.
            filter_frames_with_camera_order: Drop frames missing any camera in
                ``camera_order`` so image loading always sees a complete set.
            require_image_files: Also verify camera image files exist on disk
                while filtering (slower; off by default).
            streaming: Whether every dataloader streams scene-contiguous
                frames per batch lane for stateful temporal models.
            **kwargs: Additional base datamodule configuration.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.camera_order = camera_order
        self.name_mapping = {} if name_mapping is None else dict(name_mapping)
        self.filter_frames_with_camera_order = filter_frames_with_camera_order
        self.require_image_files = require_image_files
        self.streaming = streaming

        def resolve_ann_file(ann_file: str) -> str:
            return ann_file if os.path.isabs(ann_file) else os.path.join(data_root, ann_file)

        self.ann_files = {
            "train": resolve_ann_file(train_ann_file),
            "val": resolve_ann_file(val_ann_file),
            "test": resolve_ann_file(test_ann_file),
            "predict": resolve_ann_file(test_ann_file),
        }

    def _create_dataloader(self, split: str):
        """Create the split dataloader, streaming scenes on every split."""
        if not self.streaming:
            return super()._create_dataloader(split)
        return build_streaming_dataloader(
            getattr(self, f"{split}_dataset"),
            getattr(self, f"{split}_dataloader_cfg"),
            self.collate_fn,
            shuffle_scenes=split == "train",
        )


def _quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a scalar-first quaternion into a 3x3 rotation matrix."""
    w, x, y, z = [float(value) for value in quaternion]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.asarray(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def _build_ego_pose(sample: Mapping[str, Any]) -> np.ndarray | None:
    """Build a 4x4 ego-pose matrix from annotation fields when available."""
    matrix = sample.get("ego2global")
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float32)
    # Annotation schema variant: separate translation + quaternion fields.
    translation = sample.get("ego2global_translation")
    rotation = sample.get("ego2global_rotation")
    if translation is None or rotation is None:
        return None
    ego_pose = np.eye(4, dtype=np.float32)
    ego_pose[:3, :3] = _quaternion_to_rotation_matrix(rotation)
    ego_pose[:3, 3] = np.asarray(translation, dtype=np.float32)
    return ego_pose


class MultiviewDetection3DDataset(Dataset):
    """Base dataset for multiview 3D detection with optional lidar input.

    The dataset returns synchronized camera-lidar metadata consumed by
    transform pipelines that load images, points, and annotations on demand.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        class_names: list[str],
        camera_order: list[str],
        name_mapping: dict[str, str] | None = None,
        filter_frames_with_camera_order: bool = True,
        require_image_files: bool = False,
        dataset_transforms: Any = None,
    ) -> None:
        """Initialize the multiview detection dataset.

        Args:
            data_root: Dataset root directory.
            ann_file: Annotation file path.
            class_names: Ordered detector class names.
            camera_order: Ordered camera names expected by the model.
            name_mapping: Optional mapping from dataset labels to detector labels.
            filter_frames_with_camera_order: Drop frames missing any camera in
                ``camera_order`` (absent or null ``img_path``), so downstream
                image loading always sees a complete camera set.
            require_image_files: Also verify each camera image exists on disk
                while filtering.
            dataset_transforms: Optional transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.camera_order = camera_order
        self.name_mapping = {} if name_mapping is None else dict(name_mapping)
        self.require_image_files = require_image_files
        with open(ann_file, "rb") as file:
            data = pickle.load(file)
        self.data_infos = load_detection_data_infos(data)
        if filter_frames_with_camera_order:
            self.data_infos = self._filter_frames_with_camera_order(self.data_infos)
        self.prev_exists = self._build_prev_exists(self.data_infos)
        self.label_to_category = build_label_to_category(data.get("metainfo", {}))

    @staticmethod
    def _build_prev_exists(data_infos: list[dict[str, Any]]) -> np.ndarray:
        """Build stream-continuity flags from adjacent scene tokens."""
        prev_exists = np.zeros(len(data_infos), dtype=np.float32)
        for index in range(1, len(data_infos)):
            prev_exists[index] = np.float32(
                data_infos[index - 1]["scene_token"] == data_infos[index]["scene_token"]
            )
        return prev_exists

    def _filter_frames_with_camera_order(
        self, data_infos: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop frames that are missing any camera required by ``camera_order``.

        A frame is dropped when, for any camera in ``camera_order``, the camera
        entry is absent or its ``img_path`` is ``None`` (and, when
        ``require_image_files`` is set, when the resolved file is missing).
        """
        kept = []
        for entry in data_infos:
            images = entry.get("images", {})
            if all(
                self._camera_image_available(images.get(camera)) for camera in self.camera_order
            ):
                kept.append(entry)
        dropped = len(data_infos) - len(kept)
        if dropped:
            logger.info(
                "Filtered %d/%d frames missing one or more cameras in camera_order %s.",
                dropped,
                len(data_infos),
                list(self.camera_order),
            )
        return kept

    def _camera_image_available(self, camera_info: Mapping[str, Any] | None) -> bool:
        """Return whether one camera entry has a usable image file."""
        if camera_info is None:
            return False
        img_path = camera_info.get("img_path")
        if img_path is None:
            return False
        if self.require_image_files:
            return os.path.exists(self._resolve_path(img_path))
        return True

    def __len__(self) -> int:
        """Return the number of annotated samples.

        Returns:
            Number of samples available in the annotation file.
        """
        return len(self.data_infos)

    def scene_index_groups(self) -> list[list[int]]:
        """Group dataset indices by scene, preserving frame order.

        Returns:
            One list of scene-contiguous dataset indices per scene.
        """
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(self.data_infos):
            groups.setdefault(sample["scene_token"], []).append(index)
        return list(groups.values())

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve a path relative to the dataset root.

        Args:
            relative_path: Relative or absolute path stored in annotations.

        Returns:
            Absolute filesystem path.
        """
        normalized_path = os.path.normpath(relative_path)
        normalized_root = os.path.normpath(self.data_root)
        if os.path.isabs(normalized_path):
            return normalized_path
        if normalized_path == normalized_root or normalized_path.startswith(
            normalized_root + os.sep
        ):
            return normalized_path
        return os.path.join(normalized_root, normalized_path)

    def _resolve_lidar_path(self, sample: Mapping[str, Any]) -> str:
        """Resolve the lidar path for one sample.

        Args:
            sample: Annotation entry.

        Returns:
            Absolute lidar path.
        """
        return self._resolve_path(sample["lidar_path"])

    def _resolve_image_path(self, camera_name: str, image_path: str) -> str:
        """Resolve an image path for a camera entry.

        Args:
            camera_name: Camera identifier associated with the image entry.
            image_path: Relative or absolute image path from the annotations.

        Returns:
            Absolute filesystem path for the image.
        """
        del camera_name
        return self._resolve_path(image_path)

    def _get_camera_infos(self, sample: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Collect ordered camera metadata for one sample.

        Args:
            sample: Annotation entry for one frame.

        Returns:
            Camera metadata dictionary with resolved image paths.
        """
        images = sample.get("images", {})
        camera_infos = {}
        for camera_name in self.camera_order:
            if camera_name not in images:
                continue
            camera_info = dict(images[camera_name])
            if camera_info.get("img_path") is None:
                continue
            camera_info["img_path"] = self._resolve_image_path(camera_name, camera_info["img_path"])
            camera_infos[camera_name] = camera_info
        return camera_infos

    def _resolve_sweeps(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Resolve sweep lidar paths for one sample.

        Args:
            sample: Annotation entry for one frame.

        Returns:
            Sweep metadata list with absolute ``lidar_path`` values.
        """
        sweep_entries = []
        for sweep in sample.get("sweeps", []):
            sweep_entry = dict(sweep)
            if "lidar_path" in sweep_entry:
                sweep_entry["lidar_path"] = self._resolve_path(sweep_entry["lidar_path"])
            sweep_entries.append(sweep_entry)
        return sweep_entries

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one multiview detection metadata record.

        Args:
            index: Sample index.

        Returns:
            Metadata dictionary consumed by multiview detection transform pipelines.
        """
        sample = self.data_infos[index]
        lidar_path = self._resolve_lidar_path(sample)
        camera_infos = self._get_camera_infos(sample)
        timestamp = sample.get("timestamp")

        data_info = {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "label_to_category": self.label_to_category,
            "name_mapping": self.name_mapping,
            "images": camera_infos,
            "camera_order": self.camera_order,
            "sample_token": sample["token"],
            # Epoch-second timestamps exceed float32 resolution (256 s steps at
            # 1.7e9), so the value must stay float64 through batch collation.
            "timestamp": None if timestamp is None else np.float64(timestamp),
            "lidar_path": lidar_path,
            "num_pts_feats": int(
                sample.get("num_features", sample.get("lidar_points", {}).get("num_pts_feats", 5))
            ),
            "sweeps": self._resolve_sweeps(sample),
            "scene_token": sample["scene_token"],
            "prev_exists": self.prev_exists[index],
        }
        ego_pose = _build_ego_pose(sample)
        if ego_pose is not None:
            data_info["ego_pose"] = ego_pose
            data_info["ego_pose_inv"] = np.linalg.inv(ego_pose).astype(np.float32)
        return data_info
